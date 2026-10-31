"""Private admission-gated launcher. Payload stdin is never the control pipe."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

ADMISSION_FAILURE = 78
PAUSE_MODES = {"PAUSED_BY_USER", "WAITING_FOR_CAPACITY", "STOPPED"}


def _gate_read(descriptor, limit=129):
    """Poll anonymous pipes without a blocking reader that can stall Windows exit."""
    if os.name == "nt":
        import ctypes
        import msvcrt
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        peek = kernel.PeekNamedPipe
        peek.argtypes = [
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
        ]
        peek.restype = wintypes.BOOL
        available = wintypes.DWORD()
        if not peek(msvcrt.get_osfhandle(descriptor), None, 0, None, ctypes.byref(available), None):
            error = ctypes.get_last_error()
            if error in {109, 232}:
                return b""
            raise ctypes.WinError(error)
        if not available.value:
            return None
        return os.read(descriptor, min(available.value, limit))
    import select

    ready, _, _ = select.select([descriptor], [], [], 0)
    return os.read(descriptor, limit) if ready else None


def _alive(pid, created):
    if pid is None or created is None:
        return False
    try:
        process = psutil.Process(pid)
        return (
            abs(process.create_time() - created) < 0.01 and process.status() != psutil.STATUS_ZOMBIE
        )
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True


def _database(root):
    return sqlite3.connect(
        (Path(root) / "jobs.sqlite3").resolve().as_uri() + "?mode=rw", uri=True, timeout=5
    )


def _phase(args, phase, payload=None, payload_created=None):
    pause = None
    with _database(args.root) as db:
        if phase == "LAUNCHING":
            db.execute("BEGIN IMMEDIATE")
            mode = db.execute("SELECT value FROM settings WHERE key='mode'").fetchone()[0]
            if mode != "RUNNING":
                phase = pause = mode
        changed = db.execute(
            "UPDATE jobs SET admission_state=?,payload_pid=?,payload_created=? "
            "WHERE id=? AND admission_token=? AND pid=? AND status='RUNNING'",
            (
                phase,
                payload.pid if payload else None,
                payload_created if payload else None,
                args.job_id,
                args.token,
                os.getpid(),
            ),
        ).rowcount
        if changed != 1:
            raise ValueError("Stale controller admission state")
    if pause is not None:
        raise ValueError("Controller mode prevents payload launch: " + pause)


def _admitted_job(args):
    pause = None
    with _database(args.root) as db:
        db.execute("BEGIN IMMEDIATE")
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT * FROM jobs WHERE id=?", (args.job_id,)).fetchone()
        identity_matches = (
            row is not None
            and _alive(row["pid"], row["created"])
            and (row["pid"] == os.getpid() or (os.name == "nt" and row["pid"] == os.getppid()))
        )
        if (
            not identity_matches
            or row["status"] != "RUNNING"
            or row["admission_state"] != "WAITING"
            or row["admission_token"] != args.token
            or not _alive(row["owner_pid"], row["owner_created"])
        ):
            raise ValueError("Worker identity was not durably admitted")
        # Windows venv redirectors create a child Python process. Bind that actual
        # wrapper before it can execute payload code, not just its launcher PID.
        mode = db.execute("SELECT value FROM settings WHERE key='mode'").fetchone()[0]
        phase = "ADMITTED" if mode == "RUNNING" else mode
        changed = db.execute(
            "UPDATE jobs SET pid=?,created=?,admission_state=? "
            "WHERE id=? AND admission_token=? AND pid=? AND created=? AND admission_state='WAITING'",
            (
                os.getpid(),
                psutil.Process().create_time(),
                phase,
                args.job_id,
                args.token,
                row["pid"],
                row["created"],
            ),
        ).rowcount
        if changed != 1:
            raise ValueError("Admission changed during worker registration")
        if mode != "RUNNING":
            pause = mode
    if pause is not None:
        raise ValueError("Controller mode prevents admission: " + pause)
    command = json.loads(row["command"])
    if not command or any(not isinstance(value, str) or "\x00" in value for value in command):
        raise ValueError("Invalid registered payload command")
    return command, row["cwd"]


def _controller_mode(root):
    with _database(root) as db:
        return db.execute("SELECT value FROM settings WHERE key='mode'").fetchone()[0]


def _terminate_owned(process, known):
    if process is not None and process.pid in known and _alive(process.pid, known[process.pid]):
        with contextlib.suppress(psutil.NoSuchProcess):
            root = psutil.Process(process.pid)
            known.update({p.pid: p.create_time() for p in root.children(recursive=True)})
    targets = []
    for pid, created in known.items():
        if _alive(pid, created):
            target = psutil.Process(pid)
            if abs(target.create_time() - created) >= 0.01:
                continue
            targets.append(target)
            with contextlib.suppress(psutil.NoSuchProcess):
                target.terminate()
    _, remaining = psutil.wait_procs(targets, timeout=2)
    for target in remaining:
        with contextlib.suppress(psutil.NoSuchProcess):
            target.kill()
    # Retain the durable wrapper until all known owned processes have stopped.
    while any(_alive(pid, created) for pid, created in known.items()):
        if process is not None:
            process.poll()
        time.sleep(0.02)


def run(args):
    if args.gate_handle is not None:
        import msvcrt

        descriptor = msvcrt.open_osfhandle(args.gate_handle, os.O_RDONLY | os.O_BINARY)
    else:
        descriptor = args.gate_fd
    os.set_inheritable(descriptor, False)
    process, known = None, {}
    with os.fdopen(descriptor, "rb", buffering=0):
        token = b""
        deadline = time.monotonic() + args.admission_timeout
        while b"\n" not in token and len(token) < 129:
            chunk = _gate_read(descriptor)
            if chunk == b"":
                break
            if chunk is not None:
                token += chunk
            elif time.monotonic() >= deadline:
                raise ValueError("Controller admission timed out")
            else:
                time.sleep(0.01)
        if token != (args.token + "\n").encode("ascii"):
            raise ValueError("Controller admission missing or stale")
        command, cwd = _admitted_job(args)
        lost_owner = threading.Event()
        stop_monitor = threading.Event()

        def watch_owner():
            while not stop_monitor.is_set():
                try:
                    if _gate_read(descriptor, 1) is not None:
                        lost_owner.set()
                        return
                except OSError:
                    lost_owner.set()
                    return
                stop_monitor.wait(0.02)

        watcher = threading.Thread(target=watch_owner, daemon=True)
        watcher.start()
        try:
            if lost_owner.is_set():
                raise ValueError("Controller disappeared before payload launch")
            _phase(args, "LAUNCHING")
            process = subprocess.Popen(command, cwd=cwd, shell=False)
            try:
                known[process.pid] = psutil.Process(process.pid).create_time()
                _phase(args, "PAYLOAD", process, known[process.pid])
            except psutil.NoSuchProcess:
                pass  # A short payload may have exited; it is reaped below.
            pause = None
            while process.poll() is None and not lost_owner.is_set():
                mode = _controller_mode(args.root)
                if mode != "RUNNING":
                    pause = mode
                    break
                with contextlib.suppress(psutil.NoSuchProcess):
                    known.update(
                        {
                            p.pid: p.create_time()
                            for p in psutil.Process(process.pid).children(recursive=True)
                        }
                    )
                time.sleep(0.02)
            code = process.poll()
            if lost_owner.is_set() and code is None:
                mode = _controller_mode(args.root)
                if mode in PAUSE_MODES:
                    pause = mode
            if lost_owner.is_set() or pause is not None:
                code = ADMISSION_FAILURE
            _terminate_owned(process, known)
            _phase(args, pause or "FINISHED")
            return code if code is not None else ADMISSION_FAILURE
        finally:
            if process is not None:
                _terminate_owned(process, known)
            stop_monitor.set()
            watcher.join(timeout=1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--token", required=True)
    gate = parser.add_mutually_exclusive_group(required=True)
    gate.add_argument("--gate-fd", type=int)
    gate.add_argument("--gate-handle", type=int)
    parser.add_argument("--admission-timeout", type=float, default=15)
    args = parser.parse_args(argv)
    if not 0 < args.admission_timeout <= 60:
        parser.error("Admission timeout must be positive and at most 60 seconds")
    try:
        code = run(args)
    except (ValueError, OSError, sqlite3.Error, psutil.Error) as exc:
        print("Controller startup failed: " + str(exc), file=sys.stderr, flush=True)
        code = ADMISSION_FAILURE
    if os.name != "nt" and code < 0:
        import signal

        signal.signal(-code, signal.SIG_DFL)
        os.kill(os.getpid(), -code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
