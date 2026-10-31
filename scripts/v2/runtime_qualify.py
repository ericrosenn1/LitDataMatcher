"""Fresh application-process extraction + pretrained retrieval under a network deny hook."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from dataclasses import asdict
from pathlib import Path

from litdatamatcher.semantic_runtime import (
    LocalSemanticRuntime,
    PretrainedSemanticIndex,
    RuntimeConfig,
    digest,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--question", help="Optional user question for fresh requirement proposals")
    parser.add_argument(
        "--decoding-mode",
        choices=("unconstrained", "source_schema", "source_index", "source_literal"),
        default="unconstrained",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--require-claims",
        action="store_true",
        help="Require a source-accepted claim, not only a question, on a declared positive source",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Preserve the previous qualification; choose a new output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    blocked = []

    def deny_network(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            blocked.append(event)
            raise PermissionError("Runtime qualification denies outgoing Python sockets/DNS")

    sys.addaudithook(deny_network)
    try:
        with socket.socket() as probe:
            probe.connect(("127.0.0.1", 9))
    except PermissionError:
        pass
    else:
        raise RuntimeError("Network denial control failed")
    report = {
        "qualification_schema": "runtime-qualification-v2.3",
        "status": "FAIL",
        "network_control": {
            "kind": "Python process audit hook",
            "blocked_probe": blocked,
            "limitation": "Does not claim OS firewall control over arbitrary native libraries.",
        },
        "scientific_calibration": "NOT_RUN; smoke qualification only",
    }
    stage = "input_and_configuration"
    try:
        if args.question and args.decoding_mode in {
            "source_schema",
            "source_index",
            "source_literal",
        }:
            raise ValueError(
                f"{args.decoding_mode} is extraction-only; omit --question or use an unconstrained runtime"
            )
        document = json.loads(args.document.read_text(encoding="utf-8-sig"))
        runtime = LocalSemanticRuntime(
            args.model_dir,
            RuntimeConfig(
                device=args.device,
                dtype="bfloat16" if args.device == "cuda" else "float32",
                decoding_mode=args.decoding_mode,
                max_new_tokens=args.max_new_tokens,
            ),
        )
        expected = runtime.inference_fingerprint(document)
        report["backend_identity"] = runtime.backend_identity()
        report["expected_identity"] = expected
        report["configuration"] = asdict(runtime.config)
        stage = "fresh_extraction"
        fresh = runtime.extract(document, args.output.parent / "runtime-cache", force_fresh=True)
        report["fresh"] = fresh
        if fresh["inference_manifest"]["fingerprint"] != expected:
            raise RuntimeError(
                "Fresh runtime/code/config identity differs from qualification input"
            )
        if fresh["inference_manifest"]["configuration"] != asdict(runtime.config):
            raise RuntimeError(
                "Fresh runtime configuration differs from qualification configuration"
            )
        stage = "cached_replay"
        replay = runtime.extract(document, args.output.parent / "runtime-cache")
        report["replay_origin"] = replay["inference_manifest"]["origin"]
        report["replay_identity"] = replay["inference_manifest"]["fingerprint"]
        report["replay_configuration"] = replay["inference_manifest"]["configuration"]
        if not fresh["claims"] and not fresh["questions"]:
            raise RuntimeError("Fresh inference produced no accepted scientific records")
        if args.require_claims and not fresh["claims"]:
            raise RuntimeError("The declared positive source produced no guard-accepted claim")
        if replay["inference_manifest"]["origin"] != "cache_replay":
            raise RuntimeError("Replay did not use validated cache")
        if (
            replay["inference_manifest"]["fingerprint"] != expected
            or runtime.inference_fingerprint(document) != expected
        ):
            raise RuntimeError("Runtime/code/config identity changed during qualification")
        if runtime.backend_identity() != report["backend_identity"]:
            raise RuntimeError("Static backend identity changed during qualification")
        if report["replay_configuration"] != report["configuration"]:
            raise RuntimeError("Replay configuration changed during qualification")
        if any(digest(fresh[key]) != digest(replay[key]) for key in ("claims", "questions")):
            raise RuntimeError("Cached scientific records differ")
        stage = "pretrained_retrieval"
        index = PretrainedSemanticIndex(args.embedding_dir).fit(
            [
                {"id": "source", "text": document["title"] + " " + document["text"]},
                {
                    "id": "synthetic-negative",
                    "text": "Astronomical observations of distant stars and planetary orbits.",
                },
            ]
        )
        retrieval = index.search(document["title"], 2)
        report.update(retrieval=retrieval, embedding_model=index.manifest)
        if retrieval[0]["id"] != "source":
            raise RuntimeError("Pretrained retrieval sanity check failed")
        if args.question:
            stage = "requirement_proposal"
            report["requirement_proposal"] = runtime.interpret_question(args.question)
        report["status"] = "PASS"
    except Exception as error:
        report["failure"] = {
            "stage": stage,
            "error_type": type(error).__name__,
            "message": str(error),
        }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    fresh = report.get("fresh", {})
    print(
        json.dumps(
            {
                "status": report["status"],
                "claims": len(fresh.get("claims", [])),
                "questions": len(fresh.get("questions", [])),
                "failure": report.get("failure"),
                "elapsed_seconds": fresh.get("inference_manifest", {}).get("elapsed_seconds"),
                "output": str(args.output),
            }
        ),
        flush=True,
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
