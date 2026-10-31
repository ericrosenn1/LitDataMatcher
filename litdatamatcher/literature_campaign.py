"""Bounded PubMed continuation using the existing durable Controller queue."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import psutil

from .controller import Controller, integration_lease
from .corpus_snapshot import file_digest
from .data_plane import atomic_json, digest
from .literature_index import _IMPLEMENTATION_SHA256, LiteratureIndex

VERSION = "pubmed-current-campaign-1.0"


def continue_materialization(
    corpus_root,
    ledger_manifest,
    database,
    root,
    *,
    chunk_seconds=1200,
    max_total_seconds=7200,
    shards_per_chunk=None,
    resume=False,
    max_chunks=None,
):
    """Run serial admitted chunks, preserving completed outputs and a total cap.

    Each attempt reserves its entire timeout before launch. Successful attempts
    refund unused time; interruption conservatively retains the reservation.
    Thus restarting a coordinator cannot reset the cumulative two-hour budget.
    A new explicit campaign root can grant another bounded continuation later.
    """
    for name, value, maximum in (
        ("chunk_seconds", chunk_seconds, 1200),
        ("max_total_seconds", max_total_seconds, 7200),
    ):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer in 1..{maximum}")
    for name, value in (("shards_per_chunk", shards_per_chunk), ("max_chunks", max_chunks)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be a positive integer")
    corpus_root, ledger_manifest, database, root = (
        Path(p).resolve() for p in (corpus_root, ledger_manifest, database, root)
    )
    if root.is_relative_to(corpus_root) or root.is_relative_to(ledger_manifest.parent):
        raise ValueError("Campaign state must be outside the protected source corpus and ledger")
    state_path = root / "CORPUS_CAMPAIGN.json"
    if state_path.exists() and not resume:
        raise ValueError("Existing campaign requires --resume")
    if not state_path.exists() and resume:
        raise ValueError("No campaign exists to resume")
    if (
        not state_path.exists()
        and root.exists()
        and any(path.name not in {"integration.lock", "owner.json"} for path in root.iterdir())
    ):
        raise ValueError("Use an empty campaign state directory")
    marker = json.loads((corpus_root / "CORPUS_ROOT.json").read_text(encoding="utf-8"))
    package = Path(__file__).parent
    implementation_files = [
        "literature_index.py",
        "literature_campaign.py",
        "pubmed_revision_ledger.py",
        "corpus_snapshot.py",
        "data_plane.py",
        "controller.py",
        "controller_worker.py",
        "strict_artifacts.py",
        "recovery_contract.py",
    ]
    implementation = {name: file_digest(package / name) for name in implementation_files}
    identity = {
        "version": VERSION,
        "corpus_root": str(corpus_root),
        "store_uuid": marker["store_uuid"],
        "ledger_manifest": str(ledger_manifest),
        "ledger_sha256": file_digest(ledger_manifest),
        "database": str(database),
        "materializer_sha256": _IMPLEMENTATION_SHA256,
        "implementation": implementation,
        "python": sys.executable,
        "chunk_seconds": chunk_seconds,
        "max_total_seconds": max_total_seconds,
        "shards_per_chunk": shards_per_chunk,
    }
    with integration_lease(root, "pubmed-corpus-campaign"):
        state = (
            json.loads(state_path.read_text(encoding="utf-8"))
            if state_path.exists()
            else {
                "schema_version": VERSION,
                "campaign_id": "corpus:" + digest(identity)[:24],
                "identity": identity,
                "attempts": {},
                "completed_chunks": {},
                "created_at": time.time(),
                "overhead_seconds": 0,
            }
        )
        if state["identity"] != identity:
            raise ValueError(
                "Campaign input, implementation, interpreter or resource identity changed"
            )
        state.update(
            status="RUNNING",
            coordinator_pid=os.getpid(),
            coordinator_created=psutil.Process().create_time(),
            coordinator_active=True,
            current_job=None,
        )
        controller = Controller(root / "controller")
        controller.set_mode("RUNNING")
        completed_this_invocation = 0

        def save():
            state["budget_charged_seconds"] = (
                sum(item["charged_seconds"] for item in state["attempts"].values())
                + state["overhead_seconds"]
            )
            state["updated_at"] = time.time()
            atomic_json(state_path, state)

        save()
        try:
            while True:
                if file_digest(ledger_manifest) != identity["ledger_sha256"] or any(
                    file_digest(package / name) != value for name, value in implementation.items()
                ):
                    raise ValueError(
                        "Pinned campaign inputs or implementation changed before admission"
                    )
                # Reconcile successful chunks even if the coordinator died after
                # the worker committed its output but before this state write.
                with integration_lease(controller.root, "corpus-campaign-recovery"):
                    controller.repair()
                jobs = controller.jobs()
                for job in jobs:
                    if job["status"] == "SUCCEEDED":
                        controller._check_outputs(job, against_receipt=True)
                        report = json.loads(
                            Path(json.loads(job["outputs"])[0]).read_text(encoding="utf-8")
                        )
                        if (
                            report["ledger_manifest_sha256"] != identity["ledger_sha256"]
                            or report["implementation_sha256_at_import"]
                            != identity["materializer_sha256"]
                        ):
                            raise ValueError("Chunk output disagrees with admitted source identity")
                        state["completed_chunks"][job["id"]] = report
                        if report["status"] != "COMPLETE" and report["newly_applied_shards"] < 1:
                            state.update(
                                status="FAILED_NO_PROGRESS",
                                current_job=job["id"],
                                error="A bounded chunk committed no new shards; increase the explicit chunk allowance after diagnosis.",
                            )
                            save()
                            return state
                    elif job["status"] in {"FAILED", "RECOVERY_REQUIRED"}:
                        state.update(
                            status=job["status"], error=job["error"], current_job=job["id"]
                        )
                        save()
                        return state
                    elif job["status"] == "RUNNING":
                        state.update(
                            status="RECOVERY_REQUIRED",
                            current_job=job["id"],
                            error="An existing admitted worker is still running; preserve it and resume after it finishes.",
                        )
                        save()
                        return state
                if database.exists():
                    with LiteratureIndex(database, read_only=True) as index:
                        state["snapshot"] = index.snapshot_identity()
                        if state["snapshot"]["status"] == "COMPLETE" and all(
                            job["status"] == "SUCCEEDED" for job in jobs
                        ):
                            if (
                                index._metadata("ledger_manifest_sha256")
                                != identity["ledger_sha256"]
                                or state["snapshot"]["corpus_store_uuid"] != identity["store_uuid"]
                            ):
                                raise ValueError(
                                    "Completed database is bound to a different source snapshot"
                                )
                            state.update(
                                status="COMPLETE",
                                current_job=None,
                                validation="CHUNK_INPUT_HASHES_AND_CONTROLLER_RECEIPTS_PASS; FULL_FTS_CHECK_SEPARATE",
                            )
                            save()
                            return state
                if max_chunks is not None and completed_this_invocation >= max_chunks:
                    state.update(status="CHUNK_LIMIT", current_job=None)
                    save()
                    return state
                remaining = max_total_seconds - state["budget_charged_seconds"]
                if controller.mode() != "RUNNING":
                    state.update(status=controller.mode(), current_job=None)
                    save()
                    return state
                pending = [job for job in jobs if job["status"] in {"QUEUED", "RUNNING"}]
                if pending:
                    job = pending[0]
                    command = json.loads(job["command"])
                    allocation = float(command[command.index("--max-seconds") + 1]) + 120
                else:
                    seconds = min(chunk_seconds, int(remaining) - 120)
                    if seconds < 1:
                        state.update(status="BUDGET_LIMIT", current_job=None)
                        save()
                        return state
                    identifier = f"corpus-{len(jobs) + 1:06d}"
                    receipt = root / "chunks" / (identifier + ".json")
                    command = [
                        sys.executable,
                        "-m",
                        "litdatamatcher.literature_index",
                        "materialize",
                        "--corpus-root",
                        str(corpus_root),
                        "--ledger-manifest",
                        str(ledger_manifest),
                        "--database",
                        str(database),
                        "--max-seconds",
                        str(seconds),
                        "--receipt-out",
                        str(receipt),
                        "--expected-ledger-sha256",
                        identity["ledger_sha256"],
                        "--expected-implementation-sha256",
                        identity["materializer_sha256"],
                    ]
                    command.extend(
                        ["--max-shards", str(shards_per_chunk)]
                        if shards_per_chunk
                        else ["--all-shards"]
                    )
                    controller.register(
                        identifier,
                        command,
                        package.parent,
                        [receipt],
                        retryable_exit_codes=(75,),
                        max_retries=2,
                        retry_backoff=5,
                    )
                    job = next(row for row in controller.jobs() if row["id"] == identifier)
                    allocation = seconds + 120
                if remaining < allocation:
                    state.update(status="BUDGET_LIMIT", current_job=job["id"])
                    save()
                    return state
                if job.get("not_before", 0) > time.time():
                    delay = min(1, job["not_before"] - time.time())
                    time.sleep(delay)
                    state["overhead_seconds"] += delay
                    save()
                    continue
                key = f"{job['id']}:{job['attempts'] + 1}"
                # Lost coordinators keep their reservation charged, including
                # when Controller later refuses an ambiguous admitted process.
                state["attempts"].setdefault(
                    key, {"charged_seconds": allocation, "status": "RESERVED"}
                )
                state.update(current_job=job["id"], status="RUNNING")
                save()
                started = time.perf_counter()
                result = controller.run_next(timeout=allocation)
                state["attempts"][key] = {
                    "charged_seconds": time.perf_counter() - started,
                    "status": result["status"],
                    "job_id": job["id"],
                }
                if result.get("executed") and result["status"] == "SUCCEEDED":
                    completed_this_invocation += 1
                elif not result.get("executed"):
                    time.sleep(0.2)
                save()
        except BaseException as exc:
            state.update(
                status="INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                error=f"{type(exc).__name__}: {exc}",
            )
            save()
            raise
        finally:
            if state["status"] != "RECOVERY_REQUIRED":
                controller.set_mode(
                    "COMPLETE"
                    if state["status"] == "COMPLETE"
                    else "PAUSED_BY_USER"
                    if state["status"] == "PAUSED_BY_USER"
                    else "STOPPED"
                )
            controller.close()
            state["coordinator_active"] = False
            save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("corpus-root", "ledger-manifest", "database", "root"):
        parser.add_argument("--" + field, required=True, type=Path)
    parser.add_argument("--chunk-seconds", type=int, default=1200)
    parser.add_argument("--max-total-seconds", type=int, default=7200)
    parser.add_argument("--shards-per-chunk", type=int)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    state = continue_materialization(
        args.corpus_root,
        args.ledger_manifest,
        args.database,
        args.root,
        chunk_seconds=args.chunk_seconds,
        max_total_seconds=args.max_total_seconds,
        shards_per_chunk=args.shards_per_chunk,
        max_chunks=args.max_chunks,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                key: state.get(key)
                for key in ("campaign_id", "status", "snapshot", "budget_charged_seconds", "error")
            },
            indent=2,
        )
    )
    return 0 if state["status"] in {"COMPLETE", "CHUNK_LIMIT", "BUDGET_LIMIT"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
