"""Synthetic transport tests; no model-quality or scientific-validity labels."""

import copy
import importlib.metadata
import json

import pytest

from litdatamatcher import literal_decoding as literal
from litdatamatcher import semantic_runtime, structured_decoding
from litdatamatcher.semantic_runtime import LocalSemanticRuntime, RuntimeConfig, validate_extraction


def document(tail='in "group A" cells'):
    return {
        "document_id": "synthetic:literal",
        "text": f"Treatment reduced IL6 {tail}. Future studies should examine durability.",
    }


def payload(source=None):
    source = source or document()
    sentences = structured_decoding.complete_source_sentences(source["text"])
    return {
        "claims": [
            {
                "quote": sentences[0],
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
        "questions": [{"quote": sentences[1]}] if len(sentences) > 1 else [],
    }


def accepts(parser, value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    for char in text:
        if char not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(char)
    return parser.can_end()


@pytest.mark.parametrize(
    "tail",
    [
        'in "group A" cells',
        r"in C:\groups\A cells",
        "in β-cell samples",
        "in cafe\u0301 cells",
        "in\nnew cells",
        "in\r\nnew cells",
        "in\tnew cells",
        "in \u0000-labelled cells",
        "in 🧪 samples",
    ],
)
def test_exact_escaped_source_literal_is_accepted(tail):
    source = document(tail)
    value = payload(source)
    assert accepts(literal.schema_parser(source), value)
    assert accepts(literal.schema_parser(source), json.dumps(value, ensure_ascii=True))
    structured_decoding.validate_source_schema(value, source)
    assert validate_extraction(value, source)["claims"]


def test_public_schema_and_original_prompt_identities_are_preserved(monkeypatch):
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "0.11.3" if name == "lm-format-enforcer" else "synthetic",
    )
    runtime = LocalSemanticRuntime.__new__(LocalSemanticRuntime)
    runtime.model_manifest = {"model_id": "synthetic", "revision": "a" * 40}
    runtime.config = RuntimeConfig(decoding_mode="source_schema")
    old = runtime.inference_fingerprint(document())
    runtime.config = RuntimeConfig(decoding_mode="source_literal")
    new = runtime.inference_fingerprint(document())
    assert old["prompt_sha256"] == new["prompt_sha256"]
    assert old["prompt_version"] == new["prompt_version"]
    assert old["schema_version"] == new["schema_version"]
    assert old["decoding"]["schema_sha256"] == new["decoding"]["schema_sha256"]
    assert old["decoding"]["instruction_sha256"] == new["decoding"]["instruction_sha256"]
    assert old["decoding"]["profile"] != new["decoding"]["profile"]
    assert old["config_sha256"] != new["config_sha256"]
    changed = runtime.inference_fingerprint(document("in other cells"))
    assert new["decoding"]["schema_sha256"] != changed["decoding"]["schema_sha256"]


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "missing",
        "many_claims",
        "many_questions",
        "invented_quote",
        "truncated_quote",
        "invented_future",
        "context",
        "comparator",
        "numeric_direction",
        "numeric_negation",
        "bad_status",
        "empty_subject",
    ],
)
def test_literal_grammar_rejects_invalid_public_schema(mutation):
    value = payload()
    claim = value["claims"][0]
    if mutation == "extra":
        claim["score"] = 1
    elif mutation == "missing":
        del claim["object"]
    elif mutation == "many_claims":
        value["claims"].append(copy.deepcopy(claim))
    elif mutation == "many_questions":
        value["questions"] *= 2
    elif mutation == "invented_quote":
        claim["quote"] = "Treatment reduced another endpoint."
    elif mutation == "truncated_quote":
        claim["quote"] = "Treatment reduced IL6 in "
    elif mutation == "invented_future":
        value["questions"][0]["quote"] = "Future studies should invent a new endpoint."
    elif mutation in {"context", "comparator"}:
        claim[mutation] = "unsupported"
    elif mutation == "numeric_direction":
        claim["direction"] = 1
    elif mutation == "numeric_negation":
        claim["negated"] = 0
    elif mutation == "bad_status":
        claim["status"] = "proven"
    else:
        claim["subject"] = ""
    assert not accepts(literal.schema_parser(document()), value)


@pytest.mark.parametrize(
    "text", ["Incomplete source", "A repeated complete sentence. A repeated complete sentence."]
)
def test_no_complete_unique_sentence_requires_empty_arrays(text):
    source = {"text": text}
    assert accepts(literal.schema_parser(source), {"claims": [], "questions": []})
    assert not accepts(literal.schema_parser(source), payload())


def test_no_future_sentence_requires_empty_question_array():
    source = {"text": 'Treatment reduced IL6 in "group A" cells.'}
    value = payload(source)
    assert accepts(literal.schema_parser(source), value)
    value["questions"] = [{"quote": "Future studies should examine durability."}]
    assert not accepts(literal.schema_parser(source), value)


def test_field_order_duplicates_trailing_content_and_whitespace_are_strict():
    source = document()
    assert not accepts(literal.schema_parser(source), '{"questions":[],"claims":[]}')
    assert not accepts(literal.schema_parser(source), '{"claims":[],"claims":[],"questions":[]}')
    assert not accepts(literal.schema_parser(source), '{"claims":[],"questions":[]}null')
    assert accepts(literal.schema_parser(source), '{ "claims" : [], "questions" : [] }')
    assert not accepts(literal.schema_parser(source), '{   "claims":[],"questions":[]}')


def test_string_shortcut_is_restricted_to_generic_roles_and_preserves_prefix_choices():
    import torch

    from tests.test_structured_decoding import _character_tokenizer

    source = document()
    tokenizer = _character_tokenizer()
    parser = literal.schema_parser(source)
    hook, _ = literal.prefix_constraint(tokenizer, source)
    sequence = tokenizer.encode("0")
    text = json.dumps(payload(), ensure_ascii=False, separators=(",", ":"))
    for char in text:
        allowed = set(hook(0, torch.tensor(sequence)))
        for token_text, token_id in tokenizer.get_vocab().items():
            if len(token_text) == 1:
                assert (token_id in allowed) == (token_text in parser.get_allowed_characters())
        token = tokenizer.encode(char)[0]
        assert token in allowed
        sequence.append(token)
        parser = parser.add_character(char)
        if text[: len(sequence) - 1].endswith('"quote":"'):
            assert parser.shortcut_key() is None
        if text[: len(sequence) - 1].endswith('"subject":"'):
            assert parser.shortcut_key()[0] == "json_freetext"
    assert parser.can_end()


def test_wrong_direction_and_nonverbatim_roles_still_fail_unchanged_guard():
    for change in ({"direction": "increase"}, {"object": "invented"}, {"negated": True}):
        value = payload()
        value["claims"][0].update(change)
        assert accepts(literal.schema_parser(document()), value)
        assert not validate_extraction(value, document())["claims"]


def test_original_quote_enum_failure_remains_in_old_profile():
    assert not accepts(structured_decoding.schema_parser(document()), payload())
    assert accepts(literal.schema_parser(document()), payload())


def test_runtime_cache_keeps_exact_output_and_does_not_share_old_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "0.11.3" if name == "lm-format-enforcer" else "synthetic",
    )
    runtime = LocalSemanticRuntime.__new__(LocalSemanticRuntime)
    runtime.model_manifest = {"model_id": "synthetic", "revision": "a" * 40, "license": "fixture"}
    runtime.config = RuntimeConfig(decoding_mode="source_literal")
    runtime._generate = lambda *args: (json.dumps(payload()), 100, 100)
    result = runtime.extract(document(), tmp_path)
    assert result["claims"][0]["quote"] == payload()["claims"][0]["quote"]
    runtime._generate = lambda *args: pytest.fail("A valid cache should replay")
    assert runtime.extract(document(), tmp_path)["inference_manifest"]["origin"] == "cache_replay"
    runtime.config = RuntimeConfig(decoding_mode="source_schema")
    with pytest.raises(pytest.fail.Exception):
        runtime.extract(document(), tmp_path)


def test_literal_profile_remains_extraction_only():
    runtime = LocalSemanticRuntime.__new__(LocalSemanticRuntime)
    runtime.config = RuntimeConfig(decoding_mode="source_literal")
    with pytest.raises(ValueError, match="extraction-only"):
        runtime._generate(document(), system_prompt="Other task")


def test_retry_caps_preserve_first_records_and_trace_validated_overflow(monkeypatch):
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "0.11.3" if name == "lm-format-enforcer" else "synthetic",
    )
    source = {
        "document_id": "synthetic:retries",
        "text": "Treatment reduced IL6. Other treatment increased IL8. "
        + " ".join(f"Future studies should examine endpoint {n}." for n in range(5)),
    }
    first = payload(source)
    first["questions"] = [{"quote": f"Future studies should examine endpoint {n}."} for n in (0, 1)]
    second = copy.deepcopy(first)
    second["claims"][0].update(
        quote="Other treatment increased IL8.",
        subject="Other treatment",
        predicate="increased",
        object="IL8",
        direction="increase",
    )
    second["questions"] = [
        {"quote": f"Future studies should examine endpoint {n}."} for n in (2, 3, 4)
    ]
    responses = iter((first, second))
    runtime = LocalSemanticRuntime.__new__(LocalSemanticRuntime)
    runtime.model_manifest = {"model_id": "synthetic", "revision": "a" * 40, "license": "fixture"}
    runtime.config = RuntimeConfig(decoding_mode="source_literal")
    runtime._generate = lambda *args: (json.dumps(next(responses)), 100, 100)
    calls = []

    def engineering_retry(value, document):
        checked = validate_extraction(value, document)
        if not calls:
            checked["rejected"].append({"kind": "fixture", "reason": "Injected transport retry"})
        calls.append(True)
        return checked

    monkeypatch.setattr(semantic_runtime, "validate_extraction", engineering_retry)
    result = runtime.extract(source)
    assert len(result["claims"]) == 1 and result["claims"][0]["quote"] == "Treatment reduced IL6."
    assert [q["text"] for q in result["questions"]] == [
        f"Future studies should examine endpoint {n}." for n in range(3)
    ]
    trace = result["decoding_trace"]["attempts"]
    assert trace[0]["raw_payload"] == first and trace[1]["raw_payload"] == second
    assert len(trace[1]["validated_overflow"]) == 3
    assert trace[0]["source_guard_rejections"]


def test_duplicate_raw_keys_are_rejected():
    with pytest.raises(ValueError):
        literal.parse_literal_json('{"claims":[],"claims":[],"questions":[]}')


def test_private_document_mutation_fails_closed_without_changing_caller(monkeypatch):
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "0.11.3" if name == "lm-format-enforcer" else "synthetic",
    )
    source = document()
    runtime = LocalSemanticRuntime.__new__(LocalSemanticRuntime)
    runtime.model_manifest = {"model_id": "synthetic", "revision": "a" * 40, "license": "fixture"}
    runtime.config = RuntimeConfig(decoding_mode="source_literal")

    def corrupt(private_source, *args):
        private_source["text"] = "A changed source sentence."
        return json.dumps(payload()), 100, 100

    runtime._generate = corrupt
    with pytest.raises(RuntimeError, match="Source snapshot changed"):
        runtime.extract(source)
    assert source == document()


def test_generation_uses_exact_original_system_and_profile_instruction(monkeypatch):
    import torch

    from tests.test_structured_decoding import _character_tokenizer

    tokenizer = _character_tokenizer()
    captured = {}

    def template(messages, **kwargs):
        captured["messages"] = messages
        return "0"

    monkeypatch.setattr(tokenizer, "apply_chat_template", template)

    class Model:
        def generate(self, **kwargs):
            captured["options"] = kwargs
            return torch.tensor(
                [[*kwargs["input_ids"][0].tolist(), *tokenizer.encode(json.dumps(payload()))]]
            )

    runtime = LocalSemanticRuntime.__new__(LocalSemanticRuntime)
    runtime.config = RuntimeConfig(decoding_mode="source_literal")
    runtime._tokenizer, runtime._model = tokenizer, Model()
    runtime._generate(document())
    assert (
        captured["messages"][0]["content"]
        == semantic_runtime.SYSTEM_PROMPT + "\n" + structured_decoding.PROFILE_INSTRUCTION
    )
    assert json.loads(captured["messages"][1]["content"].split("\n", 1)[1]) == {
        "title": "",
        "text": document()["text"],
    }
    assert callable(captured["options"]["prefix_allowed_tokens_fn"])


@pytest.mark.parametrize("control", ["\u0000", "\t", "\n", "\r", "\x1f"])
def test_unescaped_controls_in_generic_roles_are_forbidden(control):
    source = document("in" + control + "samples")
    value = json.dumps(payload(source), ensure_ascii=False)
    invalid = value.replace('"subject": "Treatment"', '"subject": "Treat' + control + 'ment"')
    assert not accepts(literal.schema_parser(source), invalid)


def test_tiny_real_cpu_generation_obeys_escaped_literal():
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, LogitsProcessor

    from tests.test_structured_decoding import _character_tokenizer

    tokenizer = _character_tokenizer()
    source = {
        "document_id": "synthetic:literal",
        "text": 'Treatment reduced IL6 in "group A" cells.',
    }
    target = json.dumps(payload(source), ensure_ascii=False, separators=(",", ":"))
    tokens = tokenizer.encode(target) + [tokenizer.eos_token_id]

    class FixturePreference(LogitsProcessor):
        def __call__(self, input_ids, scores):
            scores[:, tokens[input_ids.shape[-1] - 1]] += 1000
            return scores

    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=512,
            n_embd=16,
            n_layer=1,
            n_head=2,
            eos_token_id=1,
            pad_token_id=1,
        )
    ).eval()
    prefix, _ = literal.prefix_constraint(tokenizer, source)
    with torch.inference_mode():
        output = model.generate(
            **tokenizer("0", return_tensors="pt"),
            do_sample=False,
            max_new_tokens=len(tokens),
            prefix_allowed_tokens_fn=prefix,
            logits_processor=[FixturePreference()],
            pad_token_id=1,
        )
    text = tokenizer.decode(output[0, 1:], skip_special_tokens=True)
    assert json.loads(text) == payload(source)
    assert validate_extraction(json.loads(text), source)["claims"]
