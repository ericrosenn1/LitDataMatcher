"""Opt-in exact JSON-literal grammar for the unchanged source-extraction schema."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

from . import structured_decoding

PROFILE_VERSION = "source-literal-decoding-1.0"
GRAMMAR_VERSION = "fixed-source-json-literals-1.0"


@dataclass(frozen=True)
class _Whitespace:
    """Immutable zero-to-two JSON whitespace characters between tokens."""

    count: int = 0

    def add_character(self, char):
        if char not in self.get_allowed_characters():
            raise ValueError("JSON whitespace limit exceeded")
        return replace(self, count=self.count + 1)

    def get_allowed_characters(self):
        return " \t\r\n" if self.count < 2 else ""

    def can_end(self):
        return True

    def shortcut_key(self):
        return None

    def cache_key(self):
        return ("source-literal-whitespace", self.count)


@dataclass(frozen=True)
class _StringValue:
    """Use LMFE's generic JSON string parser only inside its quoted value.

    Removing its outside whitespace avoids ambiguous overlapping whitespace
    states in a SequenceParser. String type/length validation is unchanged.
    """

    parser: object
    started: bool = False

    def get_allowed_characters(self):
        if not self.started:
            return '"'
        return "" if self.parser.can_end() else self.parser.get_allowed_characters()

    def add_character(self, char):
        if char not in self.get_allowed_characters():
            raise ValueError("Invalid JSON string character")
        return replace(self, parser=self.parser.add_character(char), started=True)

    def can_end(self):
        return self.started and self.parser.can_end()

    def shortcut_key(self):
        return self.parser.shortcut_key() if self.started and not self.can_end() else None

    def cache_key(self):
        key = self.parser.cache_key()
        return ("source-literal-string", self.started, key) if key is not None else None


@dataclass
class _Grammar:
    """Keep public composed parser semantics and expose its active string shortcut.

    LMFE's SequenceParser only forwards a shortcut for a one-element sequence.
    Our remaining fixed fields make that condition false during role strings;
    forward only from the sole active, non-terminable branch. This preserves
    the standard token enforcer's JSON-string acceleration without changing
    allowed characters or adding a token-allowance cache.
    """

    parser: object

    @property
    def config(self):
        return self.parser.config

    @config.setter
    def config(self, value):
        self.parser.config = value

    def add_character(self, char):
        return _Grammar(self.parser.add_character(char))

    def get_allowed_characters(self):
        return self.parser.get_allowed_characters()

    def can_end(self):
        return self.parser.can_end()

    def cache_key(self):
        return None

    def shortcut_key(self):
        from lmformatenforcer import SequenceParser, UnionParser

        def active(parser):
            if isinstance(parser, SequenceParser):
                for part in parser.parsers:
                    if part.can_end():
                        if part.get_allowed_characters():
                            return None  # Multiple live continuations.
                    else:
                        return active(part)
                return None
            if isinstance(parser, UnionParser):
                keys = [active(part) for part in parser.parsers]
                return keys[0] if keys and all(key == keys[0] for key in keys) else None
            return parser.shortcut_key()

        return active(self.parser)


def backend_decoding_metadata() -> dict:
    base = structured_decoding.backend_decoding_metadata()
    return {
        "profile": PROFILE_VERSION,
        "grammar_version": GRAMMAR_VERSION,
        "library": base["library"],
        "library_version": base["library_version"],
        "instruction_sha256": base["instruction_sha256"],
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_schema_implementation_sha256": base["implementation_sha256"],
        "final_cardinality": {"claims": 1, "questions": 3, "selection": "first_guard_accepted"},
        "parser_configuration": {
            "max_consecutive_whitespaces": 2,
            "force_json_field_order": True,
            "quote_encoding": "json.dumps:ensure_ascii=True|False",
            "array_limits": "source_schema",
        },
    }


def decoding_metadata(document: dict) -> dict:
    base = structured_decoding.decoding_metadata(document)
    return {
        **backend_decoding_metadata(),
        "schema_sha256": base["schema_sha256"],
        "complete_sentence_choices": base["complete_sentence_choices"],
    }


def schema_parser(document):
    """Compose the existing schema using public immutable LMFE parsers.

    Exact enums are complete serialized JSON literals, so quote/backslash,
    control-character and Unicode escaping never passes through LMFE's enum
    string state. No source text or accepted scientific content is rewritten.
    """
    backend_decoding_metadata()
    from lmformatenforcer import (
        CharacterLevelParserConfig,
        JsonSchemaParser,
        SequenceParser,
        StringParser,
        UnionParser,
    )
    from lmformatenforcer.consts import COMPLETE_ALPHABET

    schema = structured_decoding.source_schema(document)
    alphabet = "".join(
        sorted(
            set(
                COMPLETE_ALPHABET
                + "".join(chr(n) for n in range(32, 127))
                + "".join(c for c in document["text"] if c.isprintable())
            )
        )
    )
    config = CharacterLevelParserConfig(
        alphabet=alphabet, max_consecutive_whitespaces=2, force_json_field_order=True
    )

    def literal(value):
        # Some byte-level tokens end partway through a Unicode codepoint and
        # cannot traverse LMFE's character grammar. The equivalent ASCII JSON
        # escape gives every Unicode source literal a fully reachable spelling.
        spellings = dict.fromkeys(
            json.dumps(value, ensure_ascii=ascii_only, allow_nan=False)
            for ascii_only in (True, False)
        )
        return union([StringParser(spelling) for spelling in spellings])

    def sequence(parts):
        interleaved = []
        for part in parts:
            if interleaved:
                interleaved.append(_Whitespace())
            interleaved.append(part)
        return SequenceParser(interleaved)

    def union(parts):
        return parts[0] if len(parts) == 1 else UnionParser(parts)

    def compile_field(field):
        if "enum" in field:
            return union([literal(value) for value in field["enum"]])
        if field["type"] == "object":
            parts = [StringParser("{")]
            for i, name in enumerate(field["required"]):
                if i:
                    parts.append(StringParser(","))
                parts.extend(
                    (literal(name), StringParser(":"), compile_field(field["properties"][name]))
                )
            parts.append(StringParser("}"))
            return sequence(parts)
        if field["type"] == "array":
            alternatives = []
            for length in range(field["minItems"], field["maxItems"] + 1):
                parts = [StringParser("[")]
                for i in range(length):
                    if i:
                        parts.append(StringParser(","))
                    parts.append(compile_field(field["items"]))
                parts.append(StringParser("]"))
                alternatives.append(sequence(parts))
            return union(alternatives)
        if field["type"] == "string":
            return _StringValue(JsonSchemaParser(field, config=config))
        if field["type"] == "boolean":
            return union([literal(True), literal(False)])
        if field["type"] == "null":
            return literal(None)
        raise ValueError("Unsupported source schema field: " + str(field["type"]))

    return _Grammar(SequenceParser([_Whitespace(), compile_field(schema), _Whitespace()]))


def prefix_constraint(tokenizer, document, tokenizer_data=None):
    from lmformatenforcer.integrations.transformers import (
        build_token_enforcer_tokenizer_data,
        build_transformers_prefix_allowed_tokens_fn,
    )

    if tokenizer_data is None:
        tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)
    base = build_transformers_prefix_allowed_tokens_fn(tokenizer_data, schema_parser(document))
    control_tokens = {
        token: text
        for token, text, _ in tokenizer_data.regular_tokens
        if any(ord(char) < 32 for char in text)
    }
    prompt_lengths = {}

    def prefix(batch_id, sent):
        allowed = base(batch_id, sent)
        sequence = sent.tolist()
        prompt_length = prompt_lengths.setdefault(batch_id, len(sequence))
        generated = tokenizer_data.decoder(sequence[prompt_length:])
        in_string, escaped = False, False
        for char in generated:
            if in_string and escaped:
                escaped = False
            elif in_string and char == "\\":
                escaped = True
            elif char == '"':
                in_string = not in_string

        def safe(token):
            text = control_tokens.get(token)
            if text is None:
                return True
            quoted, slash = in_string, escaped
            for char in text:
                if ord(char) < 32 and (quoted or char not in "\t\r\n"):
                    return False
                if quoted and slash:
                    slash = False
                elif quoted and char == "\\":
                    slash = True
                elif char == '"':
                    quoted = not quoted
            return True

        # LMFE's json_freetext cache includes raw control tokens. Intersect only
        # those candidates with JSON lexical validity, retaining valid external
        # whitespace and quote-transition tokens. Shared token data is unchanged.
        return [token for token in allowed if safe(token)]

    return prefix, tokenizer_data


def parse_literal_json(raw: str):
    from .strict_artifacts import strict_json_loads

    return strict_json_loads(raw)
