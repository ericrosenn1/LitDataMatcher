"""Replay saved development generation through grammar; never load model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from litdatamatcher.structured_decoding import (
    decoding_metadata,
    prefix_constraint,
    schema_parser,
    source_schema,
)


def accepts(parser, text):
    for offset, char in enumerate(text):
        if char not in parser.get_allowed_characters():
            return {"accepted": False, "rejected_offset": offset, "rejected_character": char}
        parser = parser.add_character(char)
    return {"accepted": parser.can_end()}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--document", type=Path, required=True)
    cli.add_argument("--diagnostics", type=Path, required=True)
    cli.add_argument("--tokenizer-dir", type=Path)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    if args.out.exists():
        raise ValueError("Refusing to overwrite a probe receipt")
    document = json.loads(args.document.read_text(encoding="utf-8"))
    from lmformatenforcer import JsonSchemaParser

    started = time.perf_counter()
    results = []
    for path in sorted(args.diagnostics.rglob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        # The saved generations have JSON indentation beyond the grammar's
        # whitespace preference. Normalize JSON serialization only; preserve
        # all source strings and field values to isolate the array defect.
        canonical = json.dumps(json.loads(raw), ensure_ascii=False, separators=(",", ":"))
        repaired = schema_parser(document)
        incumbent = JsonSchemaParser(source_schema(document), config=repaired.config)
        results.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "serialization": "JSON whitespace normalized; field values unchanged",
                        "incumbent_grammar": accepts(incumbent, canonical), "repaired_grammar": accepts(repaired, canonical)})
    if not results:
        raise ValueError("No preserved generation diagnostics found")
    token_result = None
    if args.tokenizer_dir:
        import torch
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
        prefix, _ = prefix_constraint(tokenizer, {"text": "Treatment reduced IL6 in human cells."})
        sequence = tokenizer.encode("audit", add_special_tokens=False)
        for token in tokenizer.encode('{"claims": [], "questions": [', add_special_tokens=False):
            if token not in prefix(0, torch.tensor(sequence)):
                raise AssertionError("Tokenizer rejected a valid empty-array prefix")
            sequence.append(token)
        allowed = prefix(0, torch.tensor(sequence))
        brace = tokenizer.encode("{", add_special_tokens=False)
        closer = tokenizer.encode("]", add_special_tokens=False)
        assert len(brace) == len(closer) == 1
        token_result = {"tokenizer_dir": str(args.tokenizer_dir), "allowed_tokens": len(allowed),
                        "opening_object_allowed": brace[0] in allowed, "array_closer_allowed": closer[0] in allowed}
        if token_result["opening_object_allowed"] or not token_result["array_closer_allowed"]:
            raise AssertionError("Tokenizer-level empty-array constraint failed")
    receipt = {"status": "PASS" if all(r["incumbent_grammar"]["accepted"] and not r["repaired_grammar"]["accepted"] for r in results) else "FAIL",
               "scope": "Development grammar replay; no model inference or scientific validity claim",
               "document": {"path": str(args.document), "sha256": hashlib.sha256(args.document.read_bytes()).hexdigest()},
               "decoding": decoding_metadata(document), "diagnostics": results,
               "tokenizer_probe": token_result, "elapsed_seconds": time.perf_counter() - started}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return int(receipt["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
