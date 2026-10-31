"""Opt-in source-bounded JSON decoding; source validation remains independent."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

PROFILE_VERSION = "source-schema-decoding-1.0"
ENFORCER_VERSION = "0.11.3"
PROFILE_INSTRUCTION = (
    "Return only the constrained JSON object. Include at most one optional claim. "
    "Copy its complete source sentence, including introductory words. "
    "Use null context and comparator. Include only explicit source future-work questions, "
    "at most three. Empty arrays remain allowed; never invent a qualifying finding."
)
FUTURE_CUE = re.compile(
    r"\b(future|(?:remains?|is) (?:unclear|unknown|unresolved|uncertain)|"
    r"not (?:yet )?known|further (?:studies|research)|warrant(?:s|ed)?)\b", re.I,
)
_ABBREVIATION = re.compile(r"(?:\b(?:Dr|Mr|Mrs|Prof|Fig|Figs|vs|al|etc)|e\.g|i\.e)\.$", re.I)


def complete_source_sentences(text: str) -> list[str]:
    """Extract exact unique complete sentence strings, without rewriting whitespace.

    This conservative selector is a generation vocabulary, not a scientific
    labeler. Ambiguous repeated text and incomplete trailing fragments are absent.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Nonempty source text required for structured decoding")
    sentences = []
    start = 0
    for boundary in re.finditer(r"[.!?](?=\s|$)", text):
        stop = boundary.end()
        if text[stop - 1] == "." and _ABBREVIATION.search(text[max(start, stop - 20):stop]):
            continue
        candidate = text[start:stop].strip()
        start = stop
        if len(candidate) < 8 or text.count(candidate) != 1:
            continue
        sentences.append(candidate)
    if len(sentences) > 128:
        raise ValueError("Structured decoding requires an explicit source window of at most 128 sentences")
    return sentences


def source_schema(document: dict) -> dict:
    sentences = complete_source_sentences(document.get("text"))
    future = [sentence for sentence in sentences if FUTURE_CUE.search(sentence)]
    role = {"type": "string", "minLength": 1, "maxLength": 160}
    fields = {
        "quote": {"type": "string", "enum": sentences},
        "subject": role, "predicate": role, "object": role,
        "direction": {"type": "string", "enum": ["unknown", "increase", "decrease", "no_change", "association"]},
        "negated": {"type": "boolean"},
        "status": {"type": "string", "enum": ["interpretation", "background", "direct_experiment"]},
        "context": {"type": "null"}, "comparator": {"type": "null"},
    }
    claim = {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}
    question = {"type": "object", "properties": {"quote": {"type": "string", "enum": future}}, "required": ["quote"], "additionalProperties": False}
    return {
        "type": "object", "additionalProperties": False, "required": ["claims", "questions"],
        "properties": {
            "claims": {"type": "array", "minItems": 0, "maxItems": 1 if sentences else 0, "items": claim if sentences else {"type": "object"}},
            "questions": {"type": "array", "minItems": 0, "maxItems": min(3, len(future)), "items": question if future else {"type": "object"}},
        },
    }


def decoding_metadata(document: dict) -> dict:
    schema = source_schema(document)
    try:
        installed = importlib.metadata.version("lm-format-enforcer")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("source_schema decoding requires lm-format-enforcer==0.11.3") from error
    if installed != ENFORCER_VERSION:
        raise RuntimeError(f"source_schema requires lm-format-enforcer=={ENFORCER_VERSION}; found {installed}")
    return {
        "profile": PROFILE_VERSION, "library": "lm-format-enforcer", "library_version": installed,
        "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "instruction_sha256": hashlib.sha256(PROFILE_INSTRUCTION.encode()).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parser_configuration": {"max_consecutive_whitespaces": 2, "force_json_field_order": True},
        "complete_sentence_choices": len(complete_source_sentences(document["text"])),
    }


def validate_source_schema(payload, document):
    from jsonschema import Draft202012Validator

    errors = sorted(Draft202012Validator(source_schema(document)).iter_errors(payload), key=lambda error: str(list(error.path)))
    if errors:
        raise ValueError("Structured output violates source schema: " + errors[0].message)


def schema_parser(document):
    """Create the pinned character-level parser for this exact source document."""
    decoding_metadata(document)  # Require the tested library version, not a silent substitute.
    from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
    from lmformatenforcer.consts import COMPLETE_ALPHABET

    alphabet = "".join(sorted(set(COMPLETE_ALPHABET + "".join(char for char in document["text"] if char.isprintable()))))
    return JsonSchemaParser(source_schema(document), config=CharacterLevelParserConfig(
        alphabet=alphabet, max_consecutive_whitespaces=2, force_json_field_order=True,
    ))


def prefix_constraint(tokenizer, document, tokenizer_data=None):
    """Build a fresh parser per generation, reusing immutable tokenizer data."""
    from lmformatenforcer.integrations.transformers import (
        build_token_enforcer_tokenizer_data,
        build_transformers_prefix_allowed_tokens_fn,
    )

    parser = schema_parser(document)
    if tokenizer_data is None:
        tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)
    return build_transformers_prefix_allowed_tokens_fn(tokenizer_data, parser), tokenizer_data
