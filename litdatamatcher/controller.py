"""Small project-local corrective controller; scheduling is a separate app action.

The OS lock protects integration across worktrees and cannot be stolen based on
elapsed time. Jobs use process creation identity, atomic results and checked exits.
Only locally registered argument arrays execute; source documents never register jobs.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psutil

from .data_plane import atomic_json, digest
from .recovery_contract import recover
from .strict_artifacts import strict_json_loads, validate_artifact


class OwnershipConflict(RuntimeError):
    pass


@contextlib.contextmanager
def integration_lease(root: str | Path, owner: str):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "integration.lock"
    handle = path.open("a+b")
    locked = False
    try:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise OwnershipConflict("Integration is already owned") from exc
        locked = True
        atomic_json(
            root / "owner.json",
            {
                "owner": owner,
                "pid": os.getpid(),
                "creation_time": psutil.Process().create_time(),
                "acquired_at": time.time(),
            },
        )
        yield
    finally:
        if locked:
            atomic_json(root / "owner.json", {"owner": None, "released_at": time.time()})
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def is_alive(pid: int | None, creation_time: float | None) -> bool:
    if pid is None or creation_time is None:
        return False
    try:
        p = psutil.Process(pid)
        return p.is_running() and abs(p.create_time() - creation_time) < 0.01
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True  # unknown identity cannot authorize takeover


def _start_gated_worker(root, job_id, admission_token, cwd, admission_timeout=15):
    """Keep payload stdin unchanged; use a separate inherited admission pipe."""
    read_fd, write_fd = os.pipe()
    options = {}
    try:
        if os.name == "nt":
            import msvcrt

            os.set_inheritable(read_fd, True)
            gate_handle = msvcrt.get_osfhandle(read_fd)
            startup = subprocess.STARTUPINFO()
            startup.lpAttributeList = {"handle_list": [gate_handle]}
            options["startupinfo"] = startup
            gate = ["--gate-handle", str(gate_handle)]
        else:
            options["pass_fds"] = (read_fd,)
            gate = ["--gate-fd", str(read_fd)]
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("controller_worker.py")),
                "--root",
                str(root),
                "--job-id",
                job_id,
                "--token",
                admission_token,
                "--admission-timeout",
                str(admission_timeout),
                *gate,
            ],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False,
            **options,
        )
        writer = os.fdopen(write_fd, "wb", buffering=0)
        write_fd = None
        return process, writer
    finally:
        os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)


class Controller:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "jobs.sqlite3", timeout=20)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""PRAGMA journal_mode=WAL;
          CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT);
          INSERT OR IGNORE INTO settings VALUES('mode','RUNNING');
          CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,command TEXT,cwd TEXT,
            outputs TEXT,status TEXT,attempts INTEGER DEFAULT 0,pid INTEGER,created REAL,
            owner_pid INTEGER,owner_created REAL,exit_code INTEGER,error TEXT,
            fingerprint TEXT,started REAL,finished REAL);
        """)
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(jobs)")}
        for name, declaration in (
            ("retry_codes", "TEXT NOT NULL DEFAULT '[]'"),
            ("max_retries", "INTEGER NOT NULL DEFAULT 0"),
            ("retry_backoff", "REAL NOT NULL DEFAULT 0"),
            ("not_before", "REAL NOT NULL DEFAULT 0"),
            ("admission_token", "TEXT"),
            ("admission_state", "TEXT"),
            ("payload_pid", "INTEGER"),
            ("payload_created", "REAL"),
        ):
            if name not in columns:
                self.db.execute(f"ALTER TABLE jobs ADD COLUMN {name} {declaration}")
        self.db.commit()

    def close(self):
        self.db.close()

    def mode(self) -> str:
        return self.db.execute("SELECT value FROM settings WHERE key='mode'").fetchone()[0]

    def set_mode(self, mode: str):
        if mode not in {"RUNNING", "PAUSED_BY_USER", "WAITING_FOR_CAPACITY", "COMPLETE", "STOPPED"}:
            raise ValueError("Unknown controller mode")
        with self.db:
            self.db.execute("UPDATE settings SET value=? WHERE key='mode'", (mode,))

    def register(
        self,
        job_id: str,
        command: list[str],
        cwd: str | Path,
        outputs: list[str | Path],
        *,
        retryable_exit_codes=(),
        max_retries: int = 0,
        retry_backoff: float = 0,
    ) -> None:
        if not command or any(not isinstance(x, str) or "\x00" in x for x in command):
            raise ValueError("Job needs a literal argument array")
        if type(max_retries) is not int or not 0 <= max_retries <= 2:
            raise ValueError("max_retries must be between zero and two")
        if not isinstance(retry_backoff, (int, float)) or not 0 <= retry_backoff <= 300:
            raise ValueError("retry_backoff must be between zero and 300 seconds")
        retryable_exit_codes = list(retryable_exit_codes)
        if any(type(code) is not int or code == 0 for code in retryable_exit_codes):
            raise ValueError("Only explicit nonzero integer exit codes can be retried")
        retry_codes = sorted(set(retryable_exit_codes))
        cwd = str(Path(cwd).resolve(strict=True))
        outputs = [
            str((Path(p) if Path(p).is_absolute() else Path(cwd) / p).resolve()) for p in outputs
        ]
        fp = digest([command, cwd, outputs])
        if retry_codes or max_retries or retry_backoff:
            fp = digest([command, cwd, outputs, retry_codes, max_retries, retry_backoff])
        with self.db:
            old = self.db.execute("SELECT fingerprint FROM jobs WHERE id=?", (job_id,)).fetchone()
            if old and old[0] != fp:
                raise ValueError("Job ID collision with changed command/configuration")
            self.db.execute(
                """INSERT OR IGNORE INTO jobs(id,command,cwd,outputs,status,fingerprint,retry_codes,max_retries,retry_backoff)
              VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    job_id,
                    json.dumps(command),
                    cwd,
                    json.dumps(outputs),
                    "QUEUED",
                    fp,
                    json.dumps(retry_codes),
                    max_retries,
                    retry_backoff,
                ),
            )

    def jobs(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM jobs ORDER BY id")]

    def _receipt(self, job_id):
        return self.root / "receipts" / (hashlib.sha256(job_id.encode()).hexdigest() + ".json")

    def _check_outputs(self, job: dict, against_receipt=False) -> list[dict]:
        rows = []
        for value in json.loads(job["outputs"]):
            p = Path(value)
            if not p.is_file() or (p.stat().st_size == 0 and p.suffix.lower() != ".jsonl"):
                raise ValueError(f"Missing/empty output {p}")
            data = p.read_bytes()
            validate_artifact(p)
            rows.append(
                {
                    "path": str(p),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                }
            )
        if against_receipt:
            receipt = strict_json_loads(self._receipt(job["id"]).read_text())
            if receipt["outputs"] != rows or receipt["fingerprint"] != job["fingerprint"]:
                raise ValueError("Output receipt mismatch")
        return rows

    def repair(self) -> list[dict]:
        """Under integration lease, requeue only abandoned/corrupt task jobs."""
        if self.mode() != "RUNNING":
            return []
        repaired = []
        for job in self.jobs():
            if job["status"] == "RUNNING":
                if (
                    is_alive(job["pid"], job["created"])
                    or is_alive(job["owner_pid"], job["owner_created"])
                    or is_alive(job["payload_pid"], job["payload_created"])
                ):
                    continue
                if job["admission_state"] in {"LAUNCHING", "PAYLOAD"}:
                    with self.db:
                        self.db.execute(
                            "UPDATE jobs SET status='RECOVERY_REQUIRED',error=? WHERE id=?",
                            (
                                "Worker died after admission without confirming payload-tree shutdown",
                                job["id"],
                            ),
                        )
                    repaired.append(
                        {
                            "job_id": job["id"],
                            "reason": "Ambiguous admitted payload; automatic requeue refused",
                        }
                    )
                    continue
                reason = "Abandoned task process and runner"
            elif job["status"] == "SUCCEEDED":
                try:
                    self._check_outputs(job, against_receipt=True)
                    continue
                except (ValueError, OSError, KeyError) as exc:
                    reason = str(exc)
            else:
                continue
            # Preserve artifacts for diagnosis. Never affect paths not declared by this job.
            quarantine = (
                self.root
                / "quarantine"
                / f"{hashlib.sha256(job['id'].encode()).hexdigest()}-{time.time_ns()}"
            )
            for p in map(Path, json.loads(job["outputs"])):
                if p.is_file():
                    quarantine.mkdir(parents=True, exist_ok=True)
                    # Copy avoids moving a path used by another registered job.
                    import shutil

                    shutil.copyfile(p, quarantine / p.name)
            with self.db:
                self.db.execute(
                    "UPDATE jobs SET status='QUEUED',error=?,pid=NULL,created=NULL WHERE id=?",
                    (reason, job["id"]),
                )
            repaired.append({"job_id": job["id"], "reason": reason})
        return repaired

    def run_next(self, timeout: float = 300, max_log_bytes: int = 2_000_000) -> dict:
        """Execute one job while retaining the shared integration/repair lock."""
        with integration_lease(self.root, "deterministic-controller"):
            if self.mode() != "RUNNING":
                return {"status": self.mode(), "executed": False}
            repaired = self.repair()
            row = self.db.execute(
                "SELECT * FROM jobs WHERE status='QUEUED' AND attempts<3 AND not_before<=? ORDER BY id LIMIT 1",
                (time.time(),),
            ).fetchone()
            if row is None:
                waiting = self.db.execute(
                    "SELECT MIN(not_before) FROM jobs WHERE status='QUEUED' AND attempts<3"
                ).fetchone()[0]
                if waiting is not None:
                    return {
                        "status": "WAITING_RETRY",
                        "executed": False,
                        "next_eligible_at": waiting,
                        "repairs": repaired,
                    }
                blocked = [j["id"] for j in self.jobs() if j["status"] == "RECOVERY_REQUIRED"]
                if blocked:
                    return {
                        "status": "RECOVERY_REQUIRED",
                        "executed": False,
                        "job_ids": blocked,
                        "repairs": repaired,
                    }
                return {"status": "IDLE", "executed": False, "repairs": repaired}
            job = dict(row)
            # Prevent a stale result from being mistaken for success after a no-op command.
            before = {}
            for p in map(Path, json.loads(job["outputs"])):
                if p.is_file():
                    before[str(p)] = (
                        p.stat().st_mtime_ns,
                        hashlib.sha256(p.read_bytes()).hexdigest(),
                    )
            logs = self.root / "logs"
            logs.mkdir(exist_ok=True)
            log = logs / (hashlib.sha256(job["id"].encode()).hexdigest() + ".log")
            started = time.time()
            admission_token = uuid.uuid4().hex
            with self.db:
                self.db.execute(
                    """UPDATE jobs SET status='RUNNING',attempts=attempts+1,
                  owner_pid=?,owner_created=?,started=?,exit_code=NULL,pid=NULL,created=NULL,
                  admission_token=?,admission_state='WAITING',payload_pid=NULL,payload_created=NULL WHERE id=?""",
                    (
                        os.getpid(),
                        psutil.Process().create_time(),
                        started,
                        admission_token,
                        job["id"],
                    ),
                )
            process = None
            gate = None
            failure = None
            try:
                # Reader threads drain both streams with a fixed-size retained diagnostic buffer.
                process, gate = _start_gated_worker(
                    self.root, job["id"], admission_token, job["cwd"]
                )
                try:
                    created = psutil.Process(process.pid).create_time()
                except psutil.NoSuchProcess:
                    created = None  # A short successful child can exit before identity inspection.
                with self.db:
                    self.db.execute(
                        "UPDATE jobs SET pid=?,created=? WHERE id=?",
                        (process.pid, created, job["id"]),
                    )
                # Commit the identity before permitting any payload code to execute.
                gate.write((admission_token + "\n").encode("ascii"))
                import threading

                drain_errors = []

                def drain():
                    retained = 0
                    try:
                        with log.open("wb") as handle:
                            while chunk := process.stdout.read(8192):
                                take = min(len(chunk), max(0, max_log_bytes - retained))
                                handle.write(chunk[:take])
                                retained += take
                    except (OSError, ValueError) as exc:
                        drain_errors.append(str(exc))

                reader = threading.Thread(target=drain, daemon=True)
                reader.start()
                while process.poll() is None:
                    if time.time() - started > timeout or self.mode() != "RUNNING":
                        failure = "Timeout" if self.mode() == "RUNNING" else self.mode()
                        if is_alive(process.pid, created):
                            owned = psutil.Process(process.pid)
                            children = owned.children(recursive=True)
                            for child in reversed(children):
                                with contextlib.suppress(psutil.NoSuchProcess):
                                    child.terminate()
                            owned.terminate()
                            _, alive = psutil.wait_procs(children + [owned], timeout=3)
                            for child in alive:
                                with contextlib.suppress(psutil.NoSuchProcess):
                                    child.kill()
                        break
                    time.sleep(0.05)
                code = process.wait(timeout=10)
                reader.join(timeout=10)
                if reader.is_alive():
                    raise RuntimeError("Output drain incomplete")
                if drain_errors:
                    raise RuntimeError("Output drain failed: " + drain_errors[0])
                if failure or code != 0:
                    raise RuntimeError(failure or f"Native exit {code}")
                outputs = self._check_outputs(job)
                for entry in outputs:
                    p = Path(entry["path"])
                    if before.get(str(p)) == (p.stat().st_mtime_ns, entry["sha256"]):
                        raise ValueError("Unchanged stale output: " + str(p))
                atomic_json(
                    self._receipt(job["id"]),
                    {
                        "fingerprint": job["fingerprint"],
                        "outputs": outputs,
                        "exit_code": code,
                        "started": started,
                        "finished": time.time(),
                    },
                )
                status, error = "SUCCEEDED", None
            except (
                OSError,
                ValueError,
                RuntimeError,
                subprocess.SubprocessError,
                psutil.Error,
                sqlite3.Error,
            ) as exc:
                code = process.returncode if process is not None else None
                status, error = (
                    (
                        "QUEUED"
                        if failure in {"PAUSED_BY_USER", "WAITING_FOR_CAPACITY", "STOPPED"}
                        else "FAILED"
                    ),
                    str(exc),
                )
                if failure is None and code in json.loads(job["retry_codes"]):
                    decision = recover(
                        {"retries": job["attempts"]},
                        "TRANSIENT",
                        writer_id=job["id"],
                        max_retries=job["max_retries"],
                    )
                    if decision["status"] == "RESUMABLE":
                        status = "RETRY_PENDING"
            finally:
                if gate is not None:
                    gate.close()
            with self.db:
                if status == "QUEUED":
                    self.db.execute("UPDATE jobs SET attempts=attempts-1 WHERE id=?", (job["id"],))
                self.db.execute(
                    "UPDATE jobs SET status=?,exit_code=?,error=?,finished=?,not_before=? WHERE id=?",
                    (
                        "QUEUED" if status == "RETRY_PENDING" else status,
                        code,
                        error,
                        time.time(),
                        time.time() + min(300, job["retry_backoff"] * 2 ** job["attempts"])
                        if status == "RETRY_PENDING"
                        else 0,
                        job["id"],
                    ),
                )
            return {
                "status": status,
                "job_id": job["id"],
                "exit_code": code,
                "error": error,
                "executed": True,
                "repairs": repaired,
            }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("action", choices=["preflight", "run-next", "pause", "resume", "stop"])
    args = p.parse_args(argv)
    c = Controller(args.root)
    try:
        if args.action == "preflight":
            result = {"mode": c.mode(), "jobs": c.jobs()}
        elif args.action == "run-next":
            result = c.run_next()
        else:
            c.set_mode(
                {"pause": "PAUSED_BY_USER", "resume": "RUNNING", "stop": "STOPPED"}[args.action]
            )
            result = {"mode": c.mode()}
        print(json.dumps(result, indent=2))
        return int(result.get("status") in {"FAILED", "RECOVERY_REQUIRED"})
    finally:
        c.close()


if __name__ == "__main__":
    raise SystemExit(main())
