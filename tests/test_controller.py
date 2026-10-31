import json
import os
import subprocess
import sys
import time

import psutil
import pytest

from litdatamatcher.controller import Controller, integration_lease, is_alive


def register(c, tmp_path, name="one"):
    out = tmp_path / f"{name}.json"
    c.register(
        name,
        [
            sys.executable,
            "-c",
            f"import pathlib;pathlib.Path({str(out)!r}).write_text('{{\"ok\":true}}')",
        ],
        tmp_path,
        [out],
    )
    return out


def test_pause_corruption_actual_repair_and_resume(tmp_path):
    c = Controller(tmp_path / "controller")
    out = register(c, tmp_path)
    c.set_mode("PAUSED_BY_USER")
    assert not c.run_next()["executed"] and not out.exists()
    c.set_mode("RUNNING")
    assert c.run_next()["status"] == "SUCCEEDED"
    assert not c.run_next()["executed"]
    out.write_text("corrupt")
    result = c.run_next()
    assert result["status"] == "SUCCEEDED" and len(result["repairs"]) == 1
    assert json.loads(out.read_text()) == {"ok": True}
    assert list((c.root / "quarantine").rglob("one.json"))
    c.close()


def test_duplicate_process_writer_is_prevented(tmp_path):
    with integration_lease(tmp_path, "parent"):
        command = [
            sys.executable,
            "-c",
            f"from litdatamatcher.controller import integration_lease;\nwith integration_lease({str(tmp_path)!r},'child'): print('INVALID')",
        ]
        p = subprocess.run(command, capture_output=True, text=True)
        assert p.returncode != 0 and "OwnershipConflict" in p.stderr
        assert "INVALID" not in p.stdout


def test_abandoned_stage_resumes_without_duplicate(tmp_path):
    c = Controller(tmp_path / "controller")
    register(c, tmp_path)
    dead = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    creation = psutil.Process(dead.pid).create_time()
    dead.kill()
    dead.wait()
    with c.db:
        c.db.execute(
            "UPDATE jobs SET status='RUNNING',pid=?,created=?,owner_pid=?,owner_created=?",
            (dead.pid, creation, dead.pid, creation),
        )
    result = c.run_next()
    assert result["status"] == "SUCCEEDED" and result["repairs"]
    assert len(c.jobs()) == 1 and c.jobs()[0]["attempts"] == 1
    assert not c.run_next()["executed"]
    c.close()


def test_nonzero_stale_output_timeout_and_capacity(tmp_path):
    c = Controller(tmp_path / "controller")
    out = tmp_path / "existing.json"
    out.write_text("{}")
    c.register("bad", [sys.executable, "-c", "raise SystemExit(7)"], tmp_path, [out])
    r = c.run_next()
    assert r["status"] == "FAILED" and r["exit_code"] == 7
    c.register("stale", [sys.executable, "-c", "pass"], tmp_path, [out])
    assert c.run_next()["status"] == "FAILED"
    c.register("timeout", [sys.executable, "-c", "import time;time.sleep(30)"], tmp_path, [])
    assert c.run_next(timeout=0.1)["status"] == "FAILED"
    c.set_mode("WAITING_FOR_CAPACITY")
    assert not c.run_next()["executed"]
    c.close()


def wait_for(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("Synthetic subprocess did not reach expected state")


@pytest.mark.parametrize("window", ["before_registration", "before_admission"])
@pytest.mark.parametrize("repeat", range(3))
def test_owner_dies_before_admission_never_executes_payload(tmp_path, window, repeat):
    script = tmp_path / "crash.py"
    script.write_text(
        """import json,os,sys
from pathlib import Path
import psutil
import litdatamatcher.controller as module
root=Path(sys.argv[1]); window=sys.argv[2]
c=module.Controller(root/'state')
c.register('one',[sys.executable,'-c',"from pathlib import Path; Path('payload.json').write_text('{}')"],root,['payload.json'])
original=module._start_gated_worker
def stopped(*args):
    process,gate=original(*args)
    (root/'launched.json').write_text(json.dumps({'pid':process.pid,'created':psutil.Process(process.pid).create_time()}))
    if window=='before_registration': os._exit(86)
    class Gate:
        def write(self,value): os._exit(86)
        def close(self): gate.close()
    return process,Gate()
module._start_gated_worker=stopped
c.run_next()
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(script), str(tmp_path), window], capture_output=True, timeout=10
    )
    assert completed.returncode == 86, completed.stderr
    launched = json.loads((tmp_path / "launched.json").read_text())
    wait_for(lambda: not is_alive(launched["pid"], launched["created"]))
    assert not (tmp_path / "payload.json").exists()
    c = Controller(tmp_path / "state")
    try:
        old_token = c.jobs()[0]["admission_token"]
        result = c.run_next()
        assert result["status"] == "SUCCEEDED" and len(result["repairs"]) == 1
        assert c.jobs()[0]["admission_token"] != old_token
        assert (tmp_path / "payload.json").read_text() == "{}"
    finally:
        c.close()


def test_original_payload_stdin_and_exit_log_semantics_are_preserved(tmp_path):
    script = tmp_path / "stdin.py"
    script.write_text(
        """import json,sys
from pathlib import Path
from litdatamatcher.controller import Controller
root=Path(sys.argv[1]); c=Controller(root/'state')
c.register('stdin',[sys.executable,'-c',"import sys; from pathlib import Path; Path('stdin.json').write_text(__import__('json').dumps({'value':sys.stdin.read()})); print('stdout-marker'); print('stderr-marker',file=sys.stderr); sys.exit(7)"],root,['stdin.json'])
print(json.dumps(c.run_next())); c.close()
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        input="original stdin\n",
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "FAILED" and result["exit_code"] == 7
    assert json.loads((tmp_path / "stdin.json").read_text())["value"] == "original stdin\n"
    log = next((tmp_path / "state/logs").glob("*.log")).read_text()
    assert "stdout-marker" in log and "stderr-marker" in log


def test_dead_ambiguous_admitted_worker_is_explicitly_not_requeued(tmp_path):
    c = Controller(tmp_path / "state")
    try:
        c.register("ambiguous", [sys.executable, "-c", "pass"], tmp_path, [])
        with c.db:
            c.db.execute("UPDATE jobs SET status='RUNNING',admission_state='LAUNCHING'")
        result = c.run_next()
        assert result["status"] == "RECOVERY_REQUIRED" and not result["executed"]
        assert c.jobs()[0]["attempts"] == 0
    finally:
        c.close()


@pytest.mark.parametrize("fault", ["stale_token", "wrong_pid", "wrong_creation", "eof", "timeout"])
def test_unadmitted_worker_fails_closed(tmp_path, fault):
    from litdatamatcher.controller import _start_gated_worker

    c = Controller(tmp_path / "state")
    c.register(
        "gate",
        [sys.executable, "-c", "from pathlib import Path; Path('forbidden').touch()"],
        tmp_path,
        [],
    )
    process, gate = _start_gated_worker(
        c.root, "gate", "fixed-token", str(tmp_path), admission_timeout=0.15
    )
    try:
        with c.db:
            c.db.execute(
                "UPDATE jobs SET status='RUNNING',admission_state='WAITING',admission_token=?,pid=?,created=?,owner_pid=?,owner_created=?",
                (
                    "other-token" if fault == "stale_token" else "fixed-token",
                    process.pid + 100000 if fault == "wrong_pid" else process.pid,
                    0 if fault == "wrong_creation" else psutil.Process(process.pid).create_time(),
                    os.getpid(),
                    psutil.Process().create_time(),
                ),
            )
        if fault == "eof":
            gate.close()
        elif fault != "timeout":
            gate.write(b"fixed-token\n")
        assert process.wait(timeout=5) == 78
        assert not (tmp_path / "forbidden").exists()
    finally:
        gate.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        c.close()


def test_failed_durable_pid_write_cannot_release_payload(tmp_path):
    import sqlite3

    c = Controller(tmp_path / "state")
    c.register(
        "database",
        [sys.executable, "-c", "from pathlib import Path; Path('forbidden').touch()"],
        tmp_path,
        [],
    )
    original = c.db

    class FailingDatabase:
        failed = False

        def __getattr__(self, name):
            return getattr(original, name)

        def __enter__(self):
            original.__enter__()
            return self

        def __exit__(self, *args):
            return original.__exit__(*args)

        def execute(self, statement, *args):
            if "UPDATE jobs SET pid=" in statement and not self.failed:
                self.failed = True
                raise sqlite3.OperationalError("Injected PID registration write failure")
            return original.execute(statement, *args)

    c.db = FailingDatabase()
    try:
        result = c.run_next()
        assert result["status"] == "FAILED" and "registration" in result["error"]
        assert not (tmp_path / "forbidden").exists()
    finally:
        c.close()


def test_owner_death_during_payload_stops_descendants_before_repair(tmp_path):
    c = Controller(tmp_path / "state")
    payload = """import json,os,subprocess,sys,time,psutil
from pathlib import Path
if Path('started.json').exists():
    Path('result.json').write_text('{}')
else:
    child=subprocess.Popen([sys.executable,'-c',"import json,os,time,psutil; from pathlib import Path; Path('grandchild.json').write_text(json.dumps({'pid':os.getpid(),'created':psutil.Process().create_time()}));time.sleep(30)"])
    Path('started.json').write_text(json.dumps({'pid':os.getpid(),'created':psutil.Process().create_time()}))
    time.sleep(30)
"""
    c.register("tree", [sys.executable, "-c", payload], tmp_path, ["result.json"])
    runner = subprocess.Popen(
        [sys.executable, "-m", "litdatamatcher.controller", "--root", str(c.root), "run-next"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    identities = []
    try:
        wait_for(
            lambda: (tmp_path / "started.json").exists() and (tmp_path / "grandchild.json").exists()
        )
        identities = [
            json.loads((tmp_path / name).read_text())
            for name in ("started.json", "grandchild.json")
        ]
        job = c.jobs()[0]
        owner = psutil.Process(job["owner_pid"])
        assert abs(owner.create_time() - job["owner_created"]) < 0.01
        owner.kill()
        runner.wait(timeout=5)
        wait_for(
            lambda: all(not is_alive(item["pid"], item["created"]) for item in identities),
            timeout=8,
        )
        wait_for(lambda: not is_alive(job["pid"], job["created"]), timeout=8)
        assert not (tmp_path / "result.json").exists()
        assert c.jobs()[0]["admission_state"] == "FINISHED"
        assert c.run_next()["status"] == "SUCCEEDED"
        assert c.jobs()[0]["attempts"] == 2
    finally:
        for item in identities:
            if is_alive(item["pid"], item["created"]):
                psutil.Process(item["pid"]).kill()
        if runner.poll() is None:
            runner.kill()
            runner.wait(timeout=5)
        c.close()


def test_owner_death_around_payload_spawn_remains_owned_until_cleanup(tmp_path):
    proxy = tmp_path / "worker_proxy.py"
    proxy.write_text(
        """import sys,time
from pathlib import Path
import litdatamatcher.controller_worker as worker
root=Path(sys.argv[sys.argv.index('--root')+1]).parent
original=worker.subprocess.Popen
def delayed(*args,**kwargs):
    (root/'at_payload_spawn').touch()
    time.sleep(1)
    return original(*args,**kwargs)
worker.subprocess.Popen=delayed
raise SystemExit(worker.main())
""",
        encoding="utf-8",
    )
    script = tmp_path / "runner.py"
    script.write_text(
        """import sys
from pathlib import Path
import litdatamatcher.controller as module
root=Path(sys.argv[1]); proxy=sys.argv[2]
original=module.subprocess.Popen
def substitute(command,*args,**kwargs):
    command=list(command)
    if len(command)>1 and Path(command[1]).name=='controller_worker.py': command[1]=proxy
    return original(command,*args,**kwargs)
module.subprocess.Popen=substitute
c=module.Controller(root/'state'); c.run_next()
""",
        encoding="utf-8",
    )
    c = Controller(tmp_path / "state")
    c.register(
        "launch",
        [
            sys.executable,
            "-c",
            "import time; from pathlib import Path; time.sleep(5); Path('forbidden').touch()",
        ],
        tmp_path,
        [],
    )
    runner = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path), str(proxy)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for(lambda: (tmp_path / "at_payload_spawn").exists())
        job = c.jobs()[0]
        assert job["admission_state"] == "LAUNCHING"
        owner = psutil.Process(job["owner_pid"])
        assert abs(owner.create_time() - job["owner_created"]) < 0.01
        owner.kill()
        runner.wait(timeout=5)
        with integration_lease(c.root, "test-observer"):
            assert c.repair() == []
        wait_for(lambda: not is_alive(job["pid"], job["created"]), timeout=8)
        assert c.jobs()[0]["admission_state"] == "FINISHED"
        assert not (tmp_path / "forbidden").exists()
        with integration_lease(c.root, "test-repair"):
            assert len(c.repair()) == 1
        assert c.jobs()[0]["status"] == "QUEUED"
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.wait(timeout=5)
        c.close()
