"""Run the public batch CLI with a measured Python socket/DNS denial control.

This is an application-process control, not an operating-system firewall. Keep
the network receipt beside the batch so a failed run retains its evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
DENIED_EVENTS = frozenset(
    {
        "socket.connect",
        "socket.getaddrinfo",
        "socket.gethostbyname",
        "socket.gethostbyaddr",
        "socket.getnameinfo",
        "socket.sendto",
        "socket.sendmsg",
    }
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--network-receipt", required=True, type=Path)
    parser.add_argument("batch_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.batch_arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not arguments:
        parser.error("Pass the batch CLI arguments after --")
    receipt_path = args.network_receipt.resolve()
    if receipt_path.exists():
        raise FileExistsError("Preserve the existing network receipt; choose a new path")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    blocked = []

    def deny_network(event, _arguments):
        if event in DENIED_EVENTS:
            blocked.append(event)
            raise PermissionError("Offline batch denies outgoing Python sockets/DNS")

    # The hook is intentionally permanent for this short-lived process.
    sys.addaudithook(deny_network)
    with socket.socket() as probe:
        try:
            probe.connect(("127.0.0.1", 9))
        except PermissionError:
            pass
        else:
            raise RuntimeError("Network denial probe unexpectedly connected")
    if blocked != ["socket.connect"]:
        raise RuntimeError("Network denial probe was not observed")
    started = time.monotonic()
    report = {
        "schema_version": "offline-batch-control-1",
        "status": "RUNNING",
        "command": ["litdatamatcher-v2", "batch", *arguments],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "network_control": {
            "kind": "Python process audit hook",
            "denied_events": sorted(DENIED_EVENTS),
            "blocked_probe": list(blocked),
            "subsequent_blocked_events": [],
            "limitation": "Does not claim OS firewall control over arbitrary native libraries.",
        },
    }
    # Exclusive creation both preserves earlier evidence and marks interruption.
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    code = 1
    try:
        # The experiment must execute this checkout, not an older installed wheel.
        sys.path.insert(0, str(SOURCE_ROOT))
        from litdatamatcher import v2

        imported = Path(v2.__file__).resolve()
        if imported != SOURCE_ROOT / "litdatamatcher/v2.py":
            raise RuntimeError("Batch implementation differs from the script's source checkout")
        report["implementation"] = {
            "path": str(imported),
            "sha256": hashlib.sha256(imported.read_bytes()).hexdigest(),
        }
        code = v2.main(["batch", *arguments])
        if type(code) is not int:
            raise TypeError("The batch CLI did not return an integer exit status")
        report["status"] = "PASS" if code == 0 else "NONZERO_BATCH_EXIT"
    except BaseException as error:
        if isinstance(error, SystemExit):
            code = (
                0 if error.code is None else int(error.code) if isinstance(error.code, int) else 1
            )
            report["status"] = "CLI_EXIT"
        elif isinstance(error, KeyboardInterrupt):
            code, report["status"] = 130, "INTERRUPTED"
        else:
            code, report["status"] = 1, "FAIL"
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["exit_code"] = code
        report["elapsed_seconds"] = time.monotonic() - started
        report["network_control"]["subsequent_blocked_events"] = blocked[1:]
        # A control PASS records CLI completion, not scientific acceptance.
        from litdatamatcher.data_plane import atomic_json

        atomic_json(receipt_path, report)
        print(
            json.dumps(
                {
                    "network_receipt": str(receipt_path),
                    "status": report["status"],
                    "subsequent_blocked_events": len(blocked) - 1,
                }
            ),
            flush=True,
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
