"""Synthetic Qwen-tokenizer prefix comparison; never loads model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from litdatamatcher import literal_decoding, structured_decoding  # noqa: E402


def fixtures():
    tails = [
        ("plain", "in group A"),
        ("quoted", 'in "group A"'),
        ("backslash", r"in C:\groups\A"),
        ("unicode", "in β-cell cafe\u0301 samples"),
        ("newline", "in\nnew samples"),
        ("crlf", "in\r\nnew samples"),
        ("tab", "in\tnew samples"),
        ("nul", "in \u0000-labelled samples"),
        ("emoji", "in 🧪 samples"),
    ]
    for label, tail in tails:
        yield label, {"text": f"Treatment reduced IL6 {tail}."}, 0, False
        if not tail.isascii():
            yield label, {"text": f"Treatment reduced IL6 {tail}."}, 0, True
    for size in (8, 128):
        source = {
            "text": " ".join(f'Treatment reduced IL6 in "group {i}" cells.' for i in range(size))
        }
        yield f"shared_prefix_{size}", source, size - 1, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; preserve previous evidence")
    import torch
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
    from transformers import AutoTokenizer

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)
    startup = time.perf_counter() - started
    outcomes = []
    for name, source, choice, ascii_only in fixtures():
        sentence = structured_decoding.complete_source_sentences(source["text"])[choice]
        value = {
            "claims": [
                {
                    "quote": sentence,
                    "subject": "Treatment",
                    "predicate": "reduced",
                    "object": "IL6",
                    "direction": "decrease",
                    "negated": False,
                    "status": "direct_experiment",
                    "context": None,
                    "comparator": None,
                }
            ],
            "questions": [],
        }
        structured_decoding.validate_source_schema(value, source)
        text = json.dumps(value, ensure_ascii=ascii_only, separators=(",", ":"))
        target = tokenizer.encode(text, add_special_tokens=False)
        for mode, module in (
            ("source_schema", structured_decoding),
            ("source_literal", literal_decoding),
        ):
            t0 = time.perf_counter()
            prefix, _ = module.prefix_constraint(tokenizer, source, tokenizer_data)
            built = time.perf_counter() - t0
            sequence = tokenizer.encode("0", add_special_tokens=False)
            t0 = time.perf_counter()
            rejected_at = None
            for index, token in enumerate(target):
                if token not in prefix(0, torch.tensor(sequence)):
                    rejected_at = index
                    break
                sequence.append(token)
            eos_allowed = rejected_at is None and tokenizer.eos_token_id in prefix(
                0, torch.tensor(sequence)
            )
            outcomes.append(
                {
                    "fixture": name,
                    "serialization": "ascii_escaped" if ascii_only else "unicode_literal",
                    "mode": mode,
                    "choices": len(structured_decoding.complete_source_sentences(source["text"])),
                    "tokens": len(target),
                    "accepted_complete_target": bool(eos_allowed),
                    "first_rejected_token": rejected_at,
                    "grammar_build_seconds": built,
                    "prefix_walk_seconds": time.perf_counter() - t0,
                }
            )
            print(json.dumps(outcomes[-1]), flush=True)
    hashes = {
        name: hashlib.sha256((args.tokenizer / name).read_bytes()).hexdigest()
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
    }
    receipt = {
        "scope": "Synthetic transport and CPU prefix cost only; no pretrained inference or scientific-quality claim",
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": hashes,
        "tokenizer_setup_seconds": startup,
        "literal_backend": literal_decoding.backend_decoding_metadata(),
        "source_schema_backend": structured_decoding.backend_decoding_metadata(),
        "outcomes": outcomes,
        "limitation": "Early-rejected old-parser walks are shorter and not throughput-comparable; 128 choices bound this fixture, not production peak cost.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"receipt": str(args.output), "startup_seconds": startup, "outcomes": outcomes},
            indent=2,
        )
    )
    if not all(
        any(
            r["accepted_complete_target"]
            for r in outcomes
            if r["mode"] == "source_literal" and r["fixture"] == name
        )
        for name in {r["fixture"] for r in outcomes}
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
