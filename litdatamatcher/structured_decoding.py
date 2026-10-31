"""Opt-in source-bounded JSON decoding; source validation remains independent."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

PROFILE_VERSION = "source-schema-decoding-1.1"
ENFORCER_VERSION = "0.11.3"
PROFILE_INSTRUCTION = (
    "Return only the constrained JSON object. Include at most one optional claim. "
    "Copy its complete source sentence, including introductory words. "
    "Use null context and comparator. Include only explicit source future-work questions, "
    "at most three. Empty arrays remain allowed; never invent a qualifying finding."
)
FUTURE_CUE = re.compile(
    r"\b(future|(?:remains?|is) (?:unclear|unknown|unresolved|uncertain)|"
    r"not (?:yet )?known|further (?:studies|research)|warrant(?:s|ed)?)\b",
    re.I,
)
_ABBREVIATION = re.compile(r"(?:\b(?:Dr|Mr|Mrs|Prof|Fig|Figs|vs|al|etc)|e\.g|i\.e)\.$", re.I)


@dataclass
class _RequiredEmptyArrayParser:
    """Intersect the LMFE grammar with required-empty root arrays.

    LMFE 0.11.3 pushes a first-item parser even for maxItems=0. Its array
    const/enum alternatives are unsupported. Track only JSON lexical context
    and disallow item starts for those fields, without altering other grammar.
    Instances follow the immutable CharacterLevelParser protocol; the wrapped
    parser still owns field order, quote enums, types, and whitespace limits.
    """

    parser: object
    empty_fields: frozenset[str]
    depth: int = 0
    in_string: bool = False
    escaped: bool = False
    root_key: str = ""
    key_buffer: str = ""
    empty_array_open: bool = False

    @property
    def config(self):
        return self.parser.config

    @config.setter
    def config(self, value):
        self.parser.config = value

    def get_allowed_characters(self):
        allowed = self.parser.get_allowed_characters()
        return (
            "".join(char for char in allowed if char in "] \t\n\r")
            if self.empty_array_open
            else allowed
        )

    def add_character(self, char):
        if self.empty_array_open and char not in "] \t\n\r":
            raise ValueError("Source schema requires an empty array")
        result = replace(self, parser=self.parser.add_character(char))
        if self.in_string:
            if self.escaped:
                result.escaped = False
            elif char == "\\":
                result.escaped = True
            elif char == '"':
                result.in_string = False
                if self.depth == 1:
                    result.root_key = json.loads('"' + self.key_buffer + '"')
            if self.depth == 1 and result.in_string:
                result.key_buffer += char
        elif char == '"':
            result.in_string = True
            result.key_buffer = ""
        elif char in "[{":
            result.depth += 1
            result.empty_array_open = (
                char == "[" and self.depth == 1 and self.root_key in self.empty_fields
            )
        elif char in "]}":
            result.depth -= 1
            result.empty_array_open = False
        return result

    def can_end(self):
        return self.parser.can_end()

    def shortcut_key(self):
        return None if self.empty_array_open else self.parser.shortcut_key()

    def cache_key(self):
        key = self.parser.cache_key()
        if key is None:
            return None
        return (
            key,
            self.empty_fields,
            self.depth,
            self.in_string,
            self.escaped,
            self.root_key,
            self.key_buffer,
            self.empty_array_open,
        )


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
        if text[stop - 1] == "." and _ABBREVIATION.search(text[max(start, stop - 20) : stop]):
            continue
        candidate = text[start:stop].strip()
        start = stop
        if len(candidate) < 8 or text.count(candidate) != 1:
            continue
        sentences.append(candidate)
    if len(sentences) > 128:
        raise ValueError(
            "Structured decoding requires an explicit source window of at most 128 sentences"
        )
    return sentences


def source_schema(document: dict) -> dict:
    sentences = complete_source_sentences(document.get("text"))
    future = [sentence for sentence in sentences if FUTURE_CUE.search(sentence)]
    role = {"type": "string", "minLength": 1, "maxLength": 160}
    fields = {
        "quote": {"type": "string", "enum": sentences},
        "subject": role,
        "predicate": role,
        "object": role,
        "direction": {
            "type": "string",
            "enum": ["unknown", "increase", "decrease", "no_change", "association"],
        },
        "negated": {"type": "boolean"},
        "status": {"type": "string", "enum": ["interpretation", "background", "direct_experiment"]},
        "context": {"type": "null"},
        "comparator": {"type": "null"},
    }
    claim = {
        "type": "object",
        "properties": fields,
        "required": list(fields),
        "additionalProperties": False,
    }
    question = {
        "type": "object",
        "properties": {"quote": {"type": "string", "enum": future}},
        "required": ["quote"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["claims", "questions"],
        "properties": {
            "claims": {
                "type": "array",
                "minItems": 0,
                "maxItems": 1 if sentences else 0,
                "items": claim if sentences else {"type": "object"},
            },
            "questions": {
                "type": "array",
                "minItems": 0,
                "maxItems": min(3, len(future)),
                "items": question if future else {"type": "object"},
            },
        },
    }


def backend_decoding_metadata() -> dict:
    """Return document-independent identity for qualification and batch reuse."""
    try:
        installed = importlib.metadata.version("lm-format-enforcer")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("source_schema decoding requires lm-format-enforcer==0.11.3") from error
    if installed != ENFORCER_VERSION:
        raise RuntimeError(
            f"source_schema requires lm-format-enforcer=={ENFORCER_VERSION}; found {installed}"
        )
    return {
        "profile": PROFILE_VERSION,
        "library": "lm-format-enforcer",
        "library_version": installed,
        "instruction_sha256": hashlib.sha256(PROFILE_INSTRUCTION.encode()).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parser_configuration": {
            "max_consecutive_whitespaces": 2,
            "force_json_field_order": True,
            "required_empty_array_guard": "root-array-intersection-1",
        },
    }


def decoding_metadata(document: dict) -> dict:
    schema = source_schema(document)
    return {
        **backend_decoding_metadata(),
        "schema_sha256": hashlib.sha256(
            json.dumps(schema, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "complete_sentence_choices": len(complete_source_sentences(document["text"])),
    }


def validate_source_schema(payload, document):
    from jsonschema import Draft202012Validator

    errors = sorted(
        Draft202012Validator(source_schema(document)).iter_errors(payload),
        key=lambda error: str(list(error.path)),
    )
    if errors:
        raise ValueError("Structured output violates source schema: " + errors[0].message)


def schema_parser(document):
    """Create the pinned character-level parser for this exact source document."""
    decoding_metadata(document)  # Require the tested library version, not a silent substitute.
    from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
    from lmformatenforcer.consts import COMPLETE_ALPHABET

    alphabet = "".join(
        sorted(
            set(
                COMPLETE_ALPHABET + "".join(char for char in document["text"] if char.isprintable())
            )
        )
    )
    schema = source_schema(document)
    parser = JsonSchemaParser(
        schema,
        config=CharacterLevelParserConfig(
            alphabet=alphabet,
            max_consecutive_whitespaces=2,
            force_json_field_order=True,
        ),
    )
    empty_fields = frozenset(
        name for name, field in schema["properties"].items() if field.get("maxItems") == 0
    )
    return _RequiredEmptyArrayParser(parser, empty_fields) if empty_fields else parser


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
