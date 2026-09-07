"""Fresh application-process extraction + pretrained retrieval under a network deny hook."""
from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

from litdatamatcher.semantic_runtime import (
    LocalSemanticRuntime,
    PretrainedSemanticIndex,
    RuntimeConfig,
    digest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--question", help="Optional user question for fresh requirement proposals")
    args = parser.parse_args()
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
    document = json.loads(args.document.read_text(encoding="utf-8-sig"))
    runtime = LocalSemanticRuntime(args.model_dir, RuntimeConfig(device=args.device,
                dtype="bfloat16" if args.device == "cuda" else "float32"))
    fresh = runtime.extract(document, args.output.parent / "runtime-cache", force_fresh=True)
    replay = runtime.extract(document, args.output.parent / "runtime-cache")
    if not fresh["claims"] and not fresh["questions"]:
        raise RuntimeError("Fresh inference produced no accepted scientific records")
    if replay["inference_manifest"]["origin"] != "cache_replay":
        raise RuntimeError("Replay did not use validated cache")
    if digest(fresh["claims"]) != digest(replay["claims"]):
        raise RuntimeError("Cached scientific claims differ")
    index = PretrainedSemanticIndex(args.embedding_dir).fit([
        {"id": "source", "text": document["title"] + " " + document["text"]},
        {"id": "synthetic-negative", "text": "Astronomical observations of distant stars and planetary orbits."},
    ])
    retrieval = index.search(document["title"], 2)
    if retrieval[0]["id"] != "source":
        raise RuntimeError("Pretrained retrieval sanity check failed")
    report = {"qualification_schema": "runtime-qualification-v2.1", "status": "PASS",
              "fresh": fresh, "replay_origin": replay["inference_manifest"]["origin"],
              "retrieval": retrieval, "embedding_model": index.manifest,
              "network_control": {"kind": "Python process audit hook", "blocked_probe": blocked,
                                  "limitation": "Does not claim OS firewall control over arbitrary native libraries."},
              "scientific_calibration": "NOT_RUN; smoke qualification only"}
    if args.question:
        report["requirement_proposal"] = runtime.interpret_question(args.question)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "claims": len(fresh["claims"]),
                      "questions": len(fresh["questions"]), "elapsed_seconds": fresh["inference_manifest"]["elapsed_seconds"],
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
