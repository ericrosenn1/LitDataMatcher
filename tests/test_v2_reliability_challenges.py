"""Independent failure injection for the lead's data plane and controller."""
import json
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
