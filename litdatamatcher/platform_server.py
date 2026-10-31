"""Single-user loopback browser/API host using the existing durable controller."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from .adapters import DATASET_ADAPTERS, LITERATURE_ADAPTERS
from .controller import Controller, OwnershipConflict, integration_lease
from .data_plane import atomic_json
from .platform_workflow import validate_request
from .strict_artifacts import strict_json_loads

JOB_ID = re.compile(r"^[a-f0-9]{32}$")


class PlatformService:
    def __init__(self, root, *, configuration=None, web_root=None, job_timeout=1800):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.queue = self.root / "controller"
        self.configuration = {**(configuration or {}), "cache_root": str(self.root / "cache")}
        self.web_root = Path(web_root).resolve() if web_root else None
        self.job_timeout = job_timeout
        self.stop = threading.Event()
        self.thread = None
        self.last_worker_error = None
        controller = Controller(self.queue)
        controller.close()

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop.clear()
        self.thread = threading.Thread(
            target=self._worker, name="litdatamatcher-controller", daemon=True
        )
        self.thread.start()

    def close(self):
        self.stop.set()
        controller = Controller(self.queue)
        controller.set_mode("PAUSED_BY_USER")
        controller.close()
        if self.thread:
            self.thread.join(timeout=15)

    def _worker(self):
        while not self.stop.is_set():
            controller = Controller(self.queue)
            try:
                result = controller.run_next(timeout=self.job_timeout)
                self.last_worker_error = None
                if not result.get("executed"):
                    self.stop.wait(0.5)
            except OwnershipConflict:
                self.stop.wait(0.5)
            except (OSError, ValueError, RuntimeError) as exc:
                self.last_worker_error = str(exc)
                self.stop.wait(1)
            finally:
                controller.close()

    def health(self):
        from .semantic_runtime import runtime_dependency_versions

        versions = runtime_dependency_versions()
        model_ready = bool(
            self.configuration.get("model_dir")
            and versions.get("torch")
            and versions.get("transformers")
            and versions.get("lm-format-enforcer") == "0.11.3"
        )
        controller = Controller(self.queue)
        try:
            return {
                "status": "ok",
                "api_version": "1",
                "mode": controller.mode(),
                "worker_active": bool(self.thread and self.thread.is_alive()),
                "worker_error": self.last_worker_error,
                "model_available": model_ready,
                "semantic_available": bool(
                    self.configuration.get("embedding_dir")
                    and versions.get("torch")
                    and versions.get("transformers")
                ),
                "runtime_dependencies": versions,
                "literature_index_available": bool(self.configuration.get("literature_index")),
                "offline": bool(self.configuration.get("offline")),
                "sources": {
                    "literature": sorted(LITERATURE_ADAPTERS),
                    "datasets": sorted(DATASET_ADAPTERS),
                },
            }
        finally:
            controller.close()

    def submit(self, raw):
        request = validate_request(raw)
        if request["extraction_mode"] == "model" and not self.health()["model_available"]:
            raise ValueError(
                "Configure a verified local extraction model before selecting model mode"
            )
        if request["ranking"] != "bm25" and not self.health()["semantic_available"]:
            raise ValueError(
                "Configure a verified local embedding model before selecting semantic ranking"
            )
        identifier = uuid.uuid4().hex
        job = self.root / "jobs" / identifier
        job.mkdir(parents=True)
        atomic_json(job / "request.json", {"request": request, "configuration": self.configuration})
        atomic_json(
            job / "progress.json",
            {"stage": "queued", "status": "QUEUED", "updated_at": time.time()},
        )
        command = [
            sys.executable,
            "-m",
            "litdatamatcher.platform_workflow",
            "--request",
            str(job / "request.json"),
            "--out",
            str(job / "run"),
            "--progress",
            str(job / "progress.json"),
        ]
        controller = Controller(self.queue)
        try:
            controller.register(
                identifier,
                command,
                Path(__file__).resolve().parent.parent,
                [job / "run/result.json", job / "run/dossier.md", job / "run/candidates.csv"],
            )
        finally:
            controller.close()
        return self.job(identifier)

    def _job_path(self, identifier):
        if not JOB_ID.fullmatch(identifier):
            raise ValueError("Invalid job identifier")
        path = self.root / "jobs" / identifier
        if not path.is_dir():
            raise FileNotFoundError("Unknown job")
        return path

    def jobs(self):
        controller = Controller(self.queue)
        try:
            return {
                "mode": controller.mode(),
                "jobs": [self._view(row) for row in controller.jobs()],
            }
        finally:
            controller.close()

    def _view(self, row):
        result = {
            key: row[key]
            for key in ("id", "status", "attempts", "started", "finished", "error", "exit_code")
        }
        progress = self.root / "jobs" / row["id"] / "progress.json"
        result["progress"] = (
            strict_json_loads(progress.read_text(encoding="utf-8")) if progress.exists() else None
        )
        return result

    def job(self, identifier):
        path = self._job_path(identifier)
        controller = Controller(self.queue)
        try:
            row = next((r for r in controller.jobs() if r["id"] == identifier), None)
            if row is None:
                raise FileNotFoundError("Unknown job")
            result = self._view(row)
            result["mode"] = controller.mode()
            result["request"] = strict_json_loads(
                (path / "request.json").read_text(encoding="utf-8")
            )["request"]
            if row["status"] == "SUCCEEDED":
                controller._check_outputs(row, against_receipt=True)
                result["result"] = strict_json_loads(
                    (path / "run/result.json").read_text(encoding="utf-8")
                )
                result["retryable"] = (
                    result["result"].get("processing_status") == "PARTIAL_RETRYABLE"
                    and row["attempts"] < 3
                )
            return result
        finally:
            controller.close()

    def mode(self, mode):
        controller = Controller(self.queue)
        try:
            controller.set_mode(mode)
            return {"mode": controller.mode()}
        finally:
            controller.close()

    def retry(self, identifier):
        self._job_path(identifier)
        with integration_lease(self.queue, "application-retry"):
            controller = Controller(self.queue)
            try:
                row = next((r for r in controller.jobs() if r["id"] == identifier), None)
                partial = False
                if row and row["status"] == "SUCCEEDED":
                    controller._check_outputs(row, against_receipt=True)
                    saved = strict_json_loads(
                        (self._job_path(identifier) / "run/result.json").read_text(encoding="utf-8")
                    )
                    partial = saved.get("processing_status") == "PARTIAL_RETRYABLE"
                if not row or (row["status"] != "FAILED" and not partial) or row["attempts"] >= 3:
                    raise ValueError(
                        "Only failed or incomplete jobs with fewer than three attempts may be retried"
                    )
                with controller.db:
                    controller.db.execute(
                        "UPDATE jobs SET status='QUEUED',error=NULL,not_before=0 WHERE id=?",
                        (identifier,),
                    )
            finally:
                controller.close()
        return self.job(identifier)

    def literature_search(self, query, limit=10):
        if not query.strip() or len(query) > 4000 or not 1 <= limit <= 20:
            raise ValueError("Provide a bounded query and a limit from 1 to 20")
        if self.configuration.get("literature_index"):
            from .literature_index import LiteratureIndex

            with LiteratureIndex(self.configuration["literature_index"], read_only=True) as index:
                index.conn.execute("BEGIN")
                return {
                    "documents": index.search(query, limit=limit),
                    "coverage": index.status(include_counts=False),
                    "snapshot": index.snapshot_identity(),
                }
        from .platform_workflow import _local_rows, _select

        rows = (
            _local_rows(self.configuration["catalog_root"], "literature.jsonl")
            if self.configuration.get("catalog_root")
            else []
        )
        return {
            "documents": _select(rows, query, limit, "document_id"),
            "coverage": {
                "source": "configured_local_catalog",
                "scan_limit": 10000,
                "notice": "Use a discovery job to search public repositories.",
            },
        }


def make_server(service, *, host="127.0.0.1", port=8765):
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("This single-user service must bind to loopback")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def _respond(self, status, body, content_type="application/json; charset=utf-8"):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; connect-src 'self' https://www.ebi.ac.uk https://eutils.ncbi.nlm.nih.gov https://api.openalex.org; img-src 'self' data:; style-src 'self' 'unsafe-inline'; font-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def _origin_allowed(self):
            port = self.server.server_port
            allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if self.headers.get("Host") not in allowed:
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://" + value for value in allowed}:
                return False
            return self.headers.get("Sec-Fetch-Site", "same-origin") in {"same-origin", "none"}

        def do_GET(self):
            self._dispatch(False)

        def do_POST(self):
            self._dispatch(True)

        def _dispatch(self, write):
            if not self._origin_allowed():
                self._respond(403, {"error": "Only same-origin local requests are accepted"})
                return
            try:
                path = urlsplit(self.path).path
                payload = None
                if write:
                    if self.headers.get("Content-Type", "").split(";")[
                        0
                    ] != "application/json" or self.headers.get("Transfer-Encoding"):
                        raise ValueError("Use a JSON request body with Content-Length")
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 1_100_000:
                        raise ValueError("Request body exceeds the size limit or is empty")
                    self.connection.settimeout(15)
                    payload = strict_json_loads(self.rfile.read(length).decode("utf-8"))
                if path == "/api/health" and not write:
                    self._respond(200, service.health())
                elif path == "/api/jobs":
                    self._respond(
                        202 if write else 200, service.submit(payload) if write else service.jobs()
                    )
                elif path == "/api/requirements" and write:
                    if not isinstance(payload, dict) or set(payload) - {"question", "requirements"}:
                        raise ValueError("Provide question and optional requirements")
                    from .question_discovery import compile_research_question

                    checked = validate_request(
                        dict(payload, literature_sources=[], dataset_sources=[])
                    )
                    self._respond(
                        200,
                        compile_research_question(
                            checked["question"], user_constraints=checked["requirements"]
                        ),
                    )
                elif path in {"/api/pause", "/api/resume"} and write:
                    self._respond(
                        200, service.mode("PAUSED_BY_USER" if path.endswith("pause") else "RUNNING")
                    )
                elif path == "/api/literature" and not write:
                    query = parse_qs(urlsplit(self.path).query)
                    self._respond(
                        200,
                        service.literature_search(
                            query.get("q", [""])[0], int(query.get("limit", ["10"])[0])
                        ),
                    )
                elif path.startswith("/api/jobs/"):
                    parts = path.split("/")
                    identifier = parts[3]
                    if len(parts) == 4 and not write:
                        self._respond(200, service.job(identifier))
                    elif len(parts) == 5 and parts[4] == "retry" and write:
                        self._respond(200, service.retry(identifier))
                    elif (
                        len(parts) == 5
                        and parts[4] in {"result.json", "dossier.md", "candidates.csv"}
                        and not write
                    ):
                        job = service.job(identifier)
                        if job["status"] != "SUCCEEDED":
                            raise ValueError("Exports become available after successful validation")
                        file = service._job_path(identifier) / "run" / parts[4]
                        self._respond(
                            200,
                            file.read_bytes(),
                            mimetypes.guess_type(file)[0] or "text/plain; charset=utf-8",
                        )
                    else:
                        raise FileNotFoundError("Unknown API route")
                elif not write and not path.startswith("/api/") and service.web_root:
                    file = (service.web_root / unquote(path).lstrip("/")).resolve()
                    if not file.is_relative_to(service.web_root):
                        raise ValueError("Invalid asset path")
                    if path == "/":
                        file = service.web_root / "index.html"
                    if not file.is_file():
                        raise FileNotFoundError("Asset not found; build the web workspace first")
                    self._respond(
                        200,
                        file.read_bytes(),
                        mimetypes.guess_type(file)[0] or "application/octet-stream",
                    )
                else:
                    raise FileNotFoundError("Unknown route")
            except FileNotFoundError as exc:
                self._respond(404, {"error": str(exc)})
            except OwnershipConflict:
                self._respond(
                    409, {"error": "A job is running; wait for it to stop before retrying"}
                )
            except (ValueError, TypeError, KeyError, UnicodeError) as exc:
                self._respond(400, {"error": str(exc)})
            except (OSError, RuntimeError) as exc:
                self._respond(500, {"error": str(exc)})

    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="New application state directory; never the old supervisor root",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--web-root", default=str(Path(__file__).resolve().parent.parent / "web/dist")
    )
    for flag in ("catalog-root", "literature-index", "model-dir", "embedding-dir"):
        parser.add_argument("--" + flag)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume this app queue after verifying existing receipts",
    )
    parser.add_argument("--job-timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    if not 30 <= args.job_timeout <= 7200:
        parser.error("Job timeout must be 30..7200 seconds")
    config = {
        key: str(Path(getattr(args, key)).resolve())
        for key in ("catalog_root", "literature_index", "model_dir", "embedding_dir")
        if getattr(args, key)
    }
    for path in config.values():
        if not Path(path).exists():
            parser.error(f"Configured input does not exist: {path}")
    config.update(device=args.device, offline=args.offline)
    service = PlatformService(
        args.root, configuration=config, web_root=args.web_root, job_timeout=args.job_timeout
    )
    if args.resume:
        service.mode("RUNNING")
    server = make_server(service, port=args.port)
    service.start()
    print(
        f"LitDataMatcher: http://127.0.0.1:{server.server_port} | state {service.root}", flush=True
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
