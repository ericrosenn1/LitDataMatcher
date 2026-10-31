"""Exercise the permanent audit hook in disposable processes, never in pytest."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/campaign_20261031/run_offline_batch.py"


def run_control(tmp_path, mode):
    receipt = tmp_path / "network.json"
    program = """
import os, runpy, socket, sys, types
module = types.ModuleType('litdatamatcher.v2')
module.__file__ = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(sys.argv[1]))), 'litdatamatcher', 'v2.py')
def main(arguments):
    assert arguments == ['batch', '--root', 'fixture', '--out', 'result']
    mode = os.environ['OFFLINE_CONTROL_TEST_MODE']
    if mode == 'network':
        socket.getaddrinfo('example.invalid', 443)
    if mode == 'gethostbyname':
        socket.gethostbyname('localhost')
    if mode == 'gethostbyaddr':
        socket.gethostbyaddr('127.0.0.1')
    if mode == 'getnameinfo':
        socket.getnameinfo(('127.0.0.1', 9), 0)
    if mode in {'sendto', 'sendmsg'}:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
            if mode == 'sendto':
                client.sendto(b'probe', ('127.0.0.1', 9))
            else:
                client.sendmsg([b'probe'], [], 0, ('127.0.0.1', 9))
    if mode == 'help':
        raise SystemExit(0)
    if mode == 'help_none':
        raise SystemExit(None)
    if mode == 'bad_return':
        return None
    return int(mode)
module.main = main
sys.modules['litdatamatcher.v2'] = module
sys.argv = [sys.argv[1], '--network-receipt', sys.argv[2], '--',
            '--root', 'fixture', '--out', 'result']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    environment = {**os.environ, "PYTHONPATH": str(ROOT), "OFFLINE_CONTROL_TEST_MODE": mode}
    result = subprocess.run(
        [sys.executable, "-c", program, str(SCRIPT), str(receipt)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return result, receipt


@pytest.mark.parametrize("code", [0, 1, 75])
def test_exit_status_and_exact_arguments_are_preserved(tmp_path, code):
    result, receipt = run_control(tmp_path, str(code))
    data = json.loads(receipt.read_text())
    assert result.returncode == code
    assert data["exit_code"] == code
    assert data["status"] == ("PASS" if code == 0 else "NONZERO_BATCH_EXIT")
    assert data["network_control"]["blocked_probe"] == ["socket.connect"]
    assert data["network_control"]["subsequent_blocked_events"] == []
    assert data["command"] == ["litdatamatcher-v2", "batch", "--root", "fixture", "--out", "result"]


@pytest.mark.parametrize(
    "mode,event",
    [
        ("network", "socket.getaddrinfo"),
        ("gethostbyname", "socket.gethostbyname"),
        ("gethostbyaddr", "socket.gethostbyaddr"),
        ("getnameinfo", "socket.getnameinfo"),
        ("sendto", "socket.sendto"),
        pytest.param(
            "sendmsg",
            "socket.sendmsg",
            marks=pytest.mark.skipif(
                not hasattr(socket.socket, "sendmsg"), reason="POSIX sendmsg is unavailable"
            ),
        ),
    ],
)
def test_actual_network_attempt_is_blocked_and_retained(tmp_path, mode, event):
    result, receipt = run_control(tmp_path, mode)
    data = json.loads(receipt.read_text())
    assert result.returncode != 0
    assert data["status"] == "FAIL"
    assert data["failure"]["type"] == "PermissionError"
    assert data["network_control"]["subsequent_blocked_events"] == [event]
    assert event in data["network_control"]["denied_events"]


def test_prior_receipt_is_preserved_before_execution(tmp_path):
    receipt = tmp_path / "network.json"
    receipt.write_bytes(b"prior evidence")
    result, _ = run_control(tmp_path, "0")
    assert result.returncode != 0
    assert receipt.read_bytes() == b"prior evidence"


@pytest.mark.parametrize("mode", ["help", "help_none"])
def test_parser_exit_is_not_reported_as_batch_completion(tmp_path, mode):
    result, receipt = run_control(tmp_path, mode)
    assert result.returncode == 0
    assert json.loads(receipt.read_text())["status"] == "CLI_EXIT"
    assert json.loads(receipt.read_text())["exit_code"] == 0


def test_invalid_cli_return_cannot_be_false_success(tmp_path):
    result, receipt = run_control(tmp_path, "bad_return")
    assert result.returncode != 0
    assert json.loads(receipt.read_text())["status"] == "FAIL"


def test_direct_script_prefers_its_source_over_foreign_pythonpath(tmp_path):
    foreign = tmp_path / "foreign"
    package = foreign / "litdatamatcher"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("raise RuntimeError('stale installation imported')")
    receipt = tmp_path / "direct-network.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--network-receipt", str(receipt), "--", "--help"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(foreign)},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(receipt.read_text())
    assert data["status"] == "CLI_EXIT"
    assert Path(data["implementation"]["path"]) == ROOT / "litdatamatcher/v2.py"
