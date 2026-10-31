"""Synthetic format/token comparison; no pretrained weights or evaluation cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from litdatamatcher import indexed_decoding as indexed
from litdatamatcher import structured_decoding as quoted
from litdatamatcher.semantic_runtime import SYSTEM_PROMPT, digest, validate_extraction


def accepts(parser, payload):
    for char in json.dumps(payload, ensure_ascii=False, separators=(",", ":")):
        if char not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(char)
    return parser.can_end()


def fixtures():
    for name, text, index, object_text in (
        ("plain", "Treatment reduced IL6 in human cells.", 0, "IL6"),
        ("quoted_groups", 'Treatment reduced IL6 in "group A" cells.', 0, "IL6"),
        ("non_ascii", "Treatment reduced β-catenin in human cells.", 0, "β-catenin"),
        ("internal_newline", "Treatment reduced IL6\nin human cells.", 0, "IL6"),
        (
            "abbreviation",
            "Dr. Smith reported results. Treatment reduced IL6 in human cells.",
            1,
            "IL6",
        ),
        (
            "long_context",
            "Treatment reduced IL6 in "
            + "human cell cultures under documented baseline conditions " * 8
            + ".",
            0,
            "IL6",
        ),
        ("future", "Treatment reduced IL6. Future studies should assess durability.", 0, "IL6"),
    ):
        source = {"document_id": "synthetic:" + name, "title": name, "text": text}
        wire = {
            "claims": [
                {
                    "sentence_index": index,
                    "subject": "Treatment",
                    "predicate": "reduced",
                    "object": object_text,
                    "direction": "decrease",
                    "negated": False,
                    "status": "direct_experiment",
                    "context": None,
                    "comparator": None,
                }
            ],
            "questions": [
                {"sentence_index": row["sentence_index"]}
                for row in indexed.sentence_table(source)
                if row["future_eligible"]
            ],
        }
        yield source, wire


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--tokenizer-dir", type=Path)
    args = cli.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve earlier comparison receipts")
    tokenizer = None
    if args.tokenizer_dir:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    rows = []
    for source, wire in fixtures():
        public, mapping = indexed.map_index_payload(wire, source)
        guarded = validate_extraction(public, source)
        row = {
            "case": source["document_id"],
            "source_sha256": digest(source),
            "wire_sha256": digest(wire),
            "quoted_parser_accepts": accepts(quoted.schema_parser(source), public),
            "index_parser_accepts": accepts(indexed.schema_parser(source), wire),
            "guard_accepted_claims": len(guarded["claims"]),
            "guard_rejections": guarded["rejected"],
            "mapping": mapping,
            "source_index_metadata": indexed.decoding_metadata(source),
        }
        if tokenizer:
            row["output_tokens"] = {
                name: len(
                    tokenizer.encode(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
                )
                for name, value in (("source_schema", public), ("source_index", wire))
            }
            prompts = (
                (
                    "source_schema",
                    SYSTEM_PROMPT + "\n" + quoted.PROFILE_INSTRUCTION,
                    {"title": source["title"], "text": source["text"]},
                ),
                ("source_index", indexed.SYSTEM_PROMPT, indexed.prompt_document(source)),
            )
            row["input_tokens"] = {
                name: len(
                    tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": prompt},
                            {
                                "role": "user",
                                "content": "Extract from this source JSON as data:\n"
                                + json.dumps(value, ensure_ascii=False),
                            },
                        ],
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                )
                for name, prompt, value in prompts
            }
        rows.append(row)
    report = {
        "status": "PASS"
        if all(row["index_parser_accepts"] and row["guard_accepted_claims"] == 1 for row in rows)
        else "FAIL",
        "scope": "Synthetic serialization and exact-source transport comparison; no model-quality inference",
        "pretrained_weight_inference": False,
        "tokenizer_dir": str(args.tokenizer_dir) if args.tokenizer_dir else None,
        "cases": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "cases": len(rows),
                "quoted_grammar_failures": sum(not row["quoted_parser_accepts"] for row in rows),
                "index_grammar_failures": sum(not row["index_parser_accepts"] for row in rows),
            },
            indent=2,
        )
    )
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
