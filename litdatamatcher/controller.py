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
import time
from pathlib import Path

import psutil

from .data_plane import atomic_json, digest


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
        self, job_id: str, command: list[str], cwd: str | Path, outputs: list[str | Path]
    ) -> None:
        if not command or any(not isinstance(x, str) or "\x00" in x for x in command):
            raise ValueError("Job needs a literal argument array")
        cwd = str(Path(cwd).resolve(strict=True))
        outputs = [
            str((Path(p) if Path(p).is_absolute() else Path(cwd) / p).resolve()) for p in outputs
        ]
        fp = digest([command, cwd, outputs])
        with self.db:
            old = self.db.execute("SELECT fingerprint FROM jobs WHERE id=?", (job_id,)).fetchone()
            if old and old[0] != fp:
                raise ValueError("Job ID collision with changed command/configuration")
            self.db.execute(
                """INSERT OR IGNORE INTO jobs(id,command,cwd,outputs,status,fingerprint)
              VALUES(?,?,?,?,?,?)""",
                (job_id, json.dumps(command), cwd, json.dumps(outputs), "QUEUED", fp),
            )

    def jobs(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM jobs ORDER BY id")]

    def _receipt(self, job_id):
        return self.root / "receipts" / (hashlib.sha256(job_id.encode()).hexdigest() + ".json")

    def _check_outputs(self, job: dict, against_receipt=False) -> list[dict]:
        rows = []
        for value in json.loads(job["outputs"]):
            p = Path(value)
            if not p.is_file() or p.stat().st_size == 0:
                raise ValueError(f"Missing/empty output {p}")
            data = p.read_bytes()
            if p.suffix == ".json":
                json.loads(
                    data,
                    parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
                )
            rows.append(
                {
                    "path": str(p),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                }
            )
        if against_receipt:
            receipt = json.loads(self._receipt(job["id"]).read_text())
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
                if is_alive(job["pid"], job["created"]) or is_alive(
                    job["owner_pid"], job["owner_created"]
                ):
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
                "SELECT * FROM jobs WHERE status='QUEUED' AND attempts<3 ORDER BY id LIMIT 1"
            ).fetchone()
            if row is None:
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
            with self.db:
                self.db.execute(
                    """UPDATE jobs SET status='RUNNING',attempts=attempts+1,
                  owner_pid=?,owner_created=?,started=?,exit_code=NULL WHERE id=?""",
                    (os.getpid(), psutil.Process().create_time(), started, job["id"]),
                )
            process = None
            failure = None
            try:
                # Reader threads drain both streams with a fixed-size retained diagnostic buffer.
                process = subprocess.Popen(
                    json.loads(job["command"]),
                    cwd=job["cwd"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=False,
                )
                created = psutil.Process(process.pid).create_time()
                with self.db:
                    self.db.execute(
                        "UPDATE jobs SET pid=?,created=? WHERE id=?",
                        (process.pid, created, job["id"]),
                    )
                import threading

                def drain():
                    retained = 0
                    with log.open("wb") as handle:
                        while chunk := process.stdout.read(8192):
                            take = min(len(chunk), max(0, max_log_bytes - retained))
                            handle.write(chunk[:take])
                            retained += take

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
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                code = process.returncode if process is not None else None
                status, error = (
                    (
                        "QUEUED"
                        if failure in {"PAUSED_BY_USER", "WAITING_FOR_CAPACITY", "STOPPED"}
                        else "FAILED"
                    ),
                    str(exc),
                )
            with self.db:
                if status == "QUEUED":
                    self.db.execute("UPDATE jobs SET attempts=attempts-1 WHERE id=?", (job["id"],))
                self.db.execute(
                    "UPDATE jobs SET status=?,exit_code=?,error=?,finished=? WHERE id=?",
                    (status, code, error, time.time(), job["id"]),
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
        return int(result.get("status") == "FAILED")
    finally:
        c.close()


if __name__ == "__main__":
    raise SystemExit(main())
