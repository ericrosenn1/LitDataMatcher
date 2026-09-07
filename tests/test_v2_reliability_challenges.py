"""Independent failure injection for the lead's data plane and controller."""
import json
import os
import subprocess
import sys
import threading
import time

import pytest

from litdatamatcher.data_plane import Catalog
from litdatamatcher.controller import Controller, integration_lease, OwnershipConflict


def test_dependency_cycle_rejected_without_invalidating_valid_parent(tmp_path):
    catalog=Catalog(tmp_path)
    try:
        catalog.upsert("document","a",{"id":"a"})
        catalog.upsert("claim","b",{"id":"b"},parents=[("document","a")])
        with pytest.raises(ValueError):
            catalog.upsert("document","a",{"id":"changed"},parents=[("claim","b")])
        assert catalog.records("claim")==[{"id":"b"}]
    finally: catalog.close()


def test_changed_derivation_invalidates_descendants_even_if_text_unchanged(tmp_path):
    catalog=Catalog(tmp_path)
    try:
        for name in ("old","new"): catalog.upsert("document",name,{"id":name})
        catalog.upsert("claim","c",{"text":"same"},parents=[("document","old")])
        catalog.upsert("match","m",{"id":"m"},parents=[("claim","c")])
        catalog.upsert("claim","c",{"text":"same"},parents=[("document","new")])
        assert catalog.records("match")==[]
    finally: catalog.close()


def test_failed_parent_change_is_transactional(tmp_path):
    catalog=Catalog(tmp_path)
    try:
        catalog.upsert("document","a",{"id":"a"})
        catalog.upsert("claim","c",{"text":"good"},parents=[("document","a")])
        with pytest.raises(ValueError): catalog.upsert("claim","c",{"text":"bad"},parents=[("document","missing")])
        assert catalog.records("claim")==[{"text":"good"}]
    finally: catalog.close()


def test_stale_source_invalidates_search_results(tmp_path):
    catalog=Catalog(tmp_path)
    try:
        catalog.upsert("document","a",{"version":1})
        catalog.upsert("claim","c",{"text":"TNF"},parents=[("document","a")],search_text="TNF")
        assert catalog.search("claim","TNF")==["c"]
        catalog.upsert("document","a",{"version":2})
        assert catalog.search("claim","TNF")==[]
    finally: catalog.close()


def test_relative_outputs_resolve_against_registered_job_cwd(tmp_path):
    controller=Controller(tmp_path/"state")
    cwd=tmp_path/"work";cwd.mkdir()
    try:
        controller.register("relative",[sys.executable,"-c","pass"],cwd,["result.json"])
        assert json.loads(controller.jobs()[0]["outputs"])==[str(cwd/"result.json")]
    finally: controller.close()


def test_second_writer_cannot_take_held_lease(tmp_path):
    with integration_lease(tmp_path,"first"):
        with pytest.raises(OwnershipConflict):
            with integration_lease(tmp_path,"second"): pytest.fail("duplicate writer")


def test_stale_success_artifact_does_not_hide_noop(tmp_path):
    output=tmp_path/"result.json";output.write_text('{"old":true}')
    controller=Controller(tmp_path/"state")
    try:
        controller.register("noop",[sys.executable,"-c","pass"],tmp_path,[output])
        result=controller.run_next()
        assert result["status"]=="FAILED"
    finally: controller.close()


def test_nonzero_native_exit_never_succeeds(tmp_path):
    controller=Controller(tmp_path/"state")
    try:
        controller.register("nonzero",[sys.executable,"-c","raise SystemExit(7)"],tmp_path,[])
        result=controller.run_next()
        assert result["status"]=="FAILED" and result["exit_code"]==7
    finally: controller.close()


def test_user_pause_during_active_job_can_resume_after_explicit_resume(tmp_path):
    output=tmp_path/"result.json"
    controller=Controller(tmp_path/"state")
    code="import time,pathlib; time.sleep(.7); pathlib.Path('result.json').write_text('{}')"
    controller.register("pausable",[sys.executable,"-c",code],tmp_path,[output])
    def pause_after_start():
        control=Controller(tmp_path/"state")
        try:
            deadline=time.monotonic()+4
            while time.monotonic()<deadline:
                if control.jobs()[0]["status"]=="RUNNING":
                    control.set_mode("PAUSED_BY_USER");return
                time.sleep(.01)
            raise AssertionError("worker never started")
        finally: control.close()
    thread=threading.Thread(target=pause_after_start);thread.start()
    try:
        controller.run_next();thread.join(timeout=5)
        assert controller.mode()=="PAUSED_BY_USER"
        assert not output.exists()
        assert controller.run_next()["executed"] is False
        controller.set_mode("RUNNING")
        resumed=controller.run_next()
        assert resumed["status"]=="SUCCEEDED"
        assert json.loads(output.read_text())=={}
    finally: thread.join(timeout=5);controller.close()


def test_actual_runner_and_worker_killed_then_repaired_once(tmp_path):
    """Kill the actual registered worker and runner, not fabricated dead metadata."""
    import psutil
    controller=Controller(tmp_path/"state")
    output=tmp_path/"result.json"
    controller.register("recover",[sys.executable,"-c","import time,pathlib; time.sleep(1); pathlib.Path('result.json').write_text('{}')"],tmp_path,[output])
    runner=subprocess.Popen([sys.executable,"-m","litdatamatcher.controller","--root",str(controller.root),"run-next"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    worker=None
    try:
        deadline=time.monotonic()+8
        while time.monotonic()<deadline:
            job=controller.jobs()[0]
            if job["status"]=="RUNNING" and job["pid"]:
                worker=psutil.Process(job["pid"])
                assert abs(worker.create_time()-job["created"])<.01
                break
            time.sleep(.02)
        assert worker is not None
        # Stop the owner first so it cannot turn the killed child into an ordinary
        # FAILED exit before the supervisor observes abandonment.
        runner.kill();runner.wait(timeout=5)
        worker.kill();worker.wait(timeout=5)
        repaired=controller.run_next()
        assert repaired["status"]=="SUCCEEDED" and len(repaired["repairs"])==1
        assert len(controller.jobs())==1 and controller.jobs()[0]["attempts"]==2
        assert json.loads(output.read_text())=={}
        assert controller.run_next()["executed"] is False
    finally:
        if runner.poll() is None: runner.kill();runner.wait(timeout=5)
        if worker is not None:
            try:
                if worker.is_running(): worker.kill();worker.wait(timeout=5)
            except psutil.NoSuchProcess: pass
        controller.close()


def test_corrupt_success_repair_preserves_diagnostic_and_reexecutes(tmp_path):
    controller=Controller(tmp_path/"state");output=tmp_path/"result.json"
    controller.register("repair",[sys.executable,"-c","import pathlib; pathlib.Path('result.json').write_text('{}')"],tmp_path,[output])
    try:
        assert controller.run_next()["status"]=="SUCCEEDED"
        output.write_text("corrupt")
        repaired=controller.run_next()
        assert repaired["status"]=="SUCCEEDED" and len(repaired["repairs"])==1
        copies=list((controller.root/"quarantine").rglob("result.json"))
        assert len(copies)==1 and copies[0].read_text()=="corrupt"
        assert json.loads(output.read_text())=={}
    finally: controller.close()


def test_cross_process_duplicate_writer_refused(tmp_path):
    with integration_lease(tmp_path,"independent-parent"):
        code="from litdatamatcher.controller import integration_lease\nwith integration_lease("+repr(str(tmp_path))+",'child'): print('DUPLICATE_WRITER')"
        child=subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,timeout=10)
        assert child.returncode!=0 and "OwnershipConflict" in child.stderr
        assert "DUPLICATE_WRITER" not in child.stdout
