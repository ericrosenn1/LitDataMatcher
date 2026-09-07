import json
import os
import subprocess
import sys
import time

import pytest

from litdatamatcher.controller import Controller, OwnershipConflict, integration_lease


def register(c,tmp_path,name='one'):
    out=tmp_path/f'{name}.json'
    c.register(name,[sys.executable,'-c',f"import pathlib;pathlib.Path({str(out)!r}).write_text('{{\"ok\":true}}')"],tmp_path,[out])
    return out


def test_pause_corruption_actual_repair_and_resume(tmp_path):
    c=Controller(tmp_path/'controller');out=register(c,tmp_path)
    c.set_mode('PAUSED_BY_USER')
    assert not c.run_next()['executed'] and not out.exists()
    c.set_mode('RUNNING');assert c.run_next()['status']=='SUCCEEDED'
    assert not c.run_next()['executed']
    out.write_text('corrupt')
    result=c.run_next()
    assert result['status']=='SUCCEEDED' and len(result['repairs'])==1
    assert json.loads(out.read_text()) == {'ok':True}
    assert list((c.root/'quarantine').rglob('one.json'))
    c.close()


def test_duplicate_process_writer_is_prevented(tmp_path):
    with integration_lease(tmp_path,'parent'):
        command=[sys.executable,'-c',f"from litdatamatcher.controller import integration_lease;\nwith integration_lease({str(tmp_path)!r},'child'): print('INVALID')"]
        p=subprocess.run(command,capture_output=True,text=True)
        assert p.returncode != 0 and 'OwnershipConflict' in p.stderr
        assert 'INVALID' not in p.stdout


def test_abandoned_stage_resumes_without_duplicate(tmp_path):
    c=Controller(tmp_path/'controller');out=register(c,tmp_path)
    dead=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    import psutil
    creation=psutil.Process(dead.pid).create_time()
    dead.kill();dead.wait()
    with c.db:
        c.db.execute("UPDATE jobs SET status='RUNNING',pid=?,created=?,owner_pid=?,owner_created=?",(dead.pid,creation,dead.pid,creation))
    result=c.run_next()
    assert result['status']=='SUCCEEDED' and result['repairs']
    assert len(c.jobs())==1 and c.jobs()[0]['attempts']==1
    assert not c.run_next()['executed']
    c.close()


def test_nonzero_stale_output_timeout_and_capacity(tmp_path):
    c=Controller(tmp_path/'controller');out=tmp_path/'existing.json';out.write_text('{}')
    c.register('bad',[sys.executable,'-c','raise SystemExit(7)'],tmp_path,[out])
    r=c.run_next();assert r['status']=='FAILED' and r['exit_code']==7
    c.register('stale',[sys.executable,'-c','pass'],tmp_path,[out])
    assert c.run_next()['status']=='FAILED'
    c.register('timeout',[sys.executable,'-c','import time;time.sleep(30)'],tmp_path,[])
    assert c.run_next(timeout=.1)['status']=='FAILED'
    c.set_mode('WAITING_FOR_CAPACITY');assert not c.run_next()['executed']
    c.close()
