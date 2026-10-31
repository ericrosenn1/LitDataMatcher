"""Optional constrained decoding tests; synthetic CPU weights are not model evidence."""

import copy
import importlib.metadata
import json
import string

import pytest

from litdatamatcher.semantic_runtime import LocalSemanticRuntime, RuntimeConfig, validate_extraction
from litdatamatcher.structured_decoding import (
    complete_source_sentences,
    decoding_metadata,
    prefix_constraint,
    schema_parser,
    source_schema,
    validate_source_schema,
)


def document():
    return {
        "document_id": "synthetic:structured",
        "text": "Notably, treatment reduced IL6 in human cells. Future studies should examine durability.",
    }


def payload():
    return {
        "claims": [
            {
                "quote": "Notably, treatment reduced IL6 in human cells.",
                "subject": "treatment",
                "predicate": "reduced",
                "object": "IL6",
                "direction": "decrease",
                "negated": False,
                "status": "direct_experiment",
                "context": None,
                "comparator": None,
            }
        ],
        "questions": [{"quote": "Future studies should examine durability."}],
    }


def test_source_choices_preserve_prefix_unicode_and_whitespace():
    text = "Dr. Smith reported results. Notably, NF-κB decreased\nwith treatment. Trailing fragment"
    choices = complete_source_sentences(text)
    assert choices == ["Dr. Smith reported results.", "Notably, NF-κB decreased\nwith treatment."]


def test_repeated_or_incomplete_sentences_are_not_usable_choices():
    assert (
        complete_source_sentences("Treatment improved sleep. Treatment improved sleep. trailing")
        == []
    )
    schema = source_schema({"text": "No complete sentence"})
    assert schema["properties"]["claims"]["maxItems"] == 0


def test_schema_bounds_outputs_and_uses_exact_sentence_enums():
    schema = source_schema(document())
    assert schema["properties"]["claims"]["maxItems"] == 1
    assert schema["properties"]["questions"]["maxItems"] == 1
    validate_source_schema(payload(), document())


@pytest.mark.parametrize(
    "mutation", ["many_claims", "context", "comparator", "prefix", "question", "extra_field"]
)
def test_schema_rejects_known_generation_failures(mutation):
    value = payload()
    if mutation == "many_claims":
        value["claims"].append(copy.deepcopy(value["claims"][0]))
    elif mutation in {"context", "comparator"}:
        value["claims"][0][mutation] = "invented context"
    elif mutation == "prefix":
        value["claims"][0]["quote"] = "treatment reduced IL6 in human cells."
    elif mutation == "question":
        value["questions"][0]["quote"] = "Future studies should examine an invented endpoint."
    else:
        value["claims"][0]["confidence"] = 1.0
    with pytest.raises(ValueError, match="source schema"):
        validate_source_schema(value, document())


def test_format_valid_hallucinated_relation_still_fails_independent_guard():
    value = payload()
    value["claims"][0]["direction"] = "increase"
    validate_source_schema(value, document())
    result = validate_extraction(value, document())
    assert result["claims"] == []
    assert result["rejected"]


def test_explicit_empty_output_remains_allowed():
    validate_source_schema({"claims": [], "questions": []}, document())


def test_config_is_opt_in_and_rejects_unknown_profile():
    assert RuntimeConfig().decoding_mode == "unconstrained"
    assert RuntimeConfig(decoding_mode="source_schema", max_new_tokens=384).max_new_tokens == 384
    with pytest.raises(ValueError, match="decoding_mode"):
        RuntimeConfig(decoding_mode="any_json")


def test_schema_hash_tracks_source_choices_and_library_pin(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.11.3")
    first = decoding_metadata(document())
    second = decoding_metadata({"text": "Treatment increased IL6 in human cells."})
    assert first["schema_sha256"] != second["schema_sha256"]
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.11.2")
    with pytest.raises(RuntimeError, match="0.11.3"):
        decoding_metadata(document())


def test_runtime_fingerprint_includes_profile_and_source_schema(monkeypatch):
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "0.11.3" if name == "lm-format-enforcer" else "synthetic-version",
    )
    runtime = LocalSemanticRuntime.__new__(LocalSemanticRuntime)
    runtime.config = RuntimeConfig(decoding_mode="source_schema")
    runtime.model_manifest = {"model_id": "synthetic-test", "revision": "0" * 40, "license": "test"}
    runtime._generate = lambda *args: (json.dumps(payload()), 10, 30)
    result = runtime.extract(document())
    metadata = result["inference_manifest"]["fingerprint"]["decoding"]
    assert metadata["mode"] == "source_schema"
    assert metadata["library_version"] == "0.11.3"
    assert metadata["schema_sha256"]
    assert result["claims"]


def test_backend_identity_is_document_independent_and_tracks_configuration(monkeypatch):
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "0.11.3" if name == "lm-format-enforcer" else "synthetic-version",
    )
    runtime = LocalSemanticRuntime.__new__(LocalSemanticRuntime)
    runtime.config = RuntimeConfig(decoding_mode="source_schema")
    runtime.model_manifest = {"model_id": "synthetic-test", "revision": "0" * 40}
    backend = runtime.backend_identity()
    assert "input_sha256" not in backend
    assert "schema_sha256" not in backend["decoding"]
    assert "complete_sentence_choices" not in backend["decoding"]
    for source in (document(), {"text": "A different complete source sentence."}):
        bound = runtime.inference_fingerprint(source)
        assert all(bound[key] == value for key, value in backend.items() if key != "decoding")
        assert all(bound["decoding"][key] == value for key, value in backend["decoding"].items())
    runtime.config = RuntimeConfig(decoding_mode="source_schema", max_new_tokens=384)
    assert runtime.backend_identity()["config_sha256"] != backend["config_sha256"]


def test_unconstrained_profile_does_not_require_optional_library(monkeypatch):
    def version(name):
        if name == "lm-format-enforcer":
            raise importlib.metadata.PackageNotFoundError(name)
        return "synthetic-version"

    monkeypatch.setattr(importlib.metadata, "version", version)
    runtime = LocalSemanticRuntime.__new__(LocalSemanticRuntime)
    runtime.config = RuntimeConfig()
    runtime.model_manifest = {"model_id": "synthetic-test", "revision": "0" * 40, "license": "test"}
    runtime._generate = lambda *args: (json.dumps(payload()), 10, 30)
    assert runtime.extract(document())["claims"]


def test_source_profile_never_silently_decodes_requirement_proposals():
    pytest.importorskip("torch")
    runtime = LocalSemanticRuntime.__new__(LocalSemanticRuntime)
    runtime.config = RuntimeConfig(decoding_mode="source_schema")
    with pytest.raises(ValueError, match="extraction-only"):
        runtime._generate(document(), system_prompt="A different proposal task")


def _accepts(parser, value):
    for char in json.dumps(value, ensure_ascii=False):
        if char not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(char)
    return parser.can_end()


def test_actual_character_parser_accepts_only_bounded_source_output():
    pytest.importorskip("lmformatenforcer")
    assert _accepts(schema_parser(document()), payload())
    value = payload()
    value["claims"][0]["context"] = "human cells"
    assert not _accepts(schema_parser(document()), value)
    value = payload()
    value["claims"][0]["quote"] = "treatment reduced IL6 in human cells."
    assert not _accepts(schema_parser(document()), value)


@pytest.mark.parametrize("field", ["claims", "questions"])
def test_actual_parser_forbids_first_item_in_required_empty_array(field):
    pytest.importorskip("lmformatenforcer")
    source = (
        {"text": "Incomplete source fragment"}
        if field == "claims"
        else {"text": "Treatment reduced IL6 in human cells."}
    )
    value = {"claims": [], "questions": []}
    value[field] = [{"quote": "Invented future research."}]
    assert not _accepts(schema_parser(source), value)
    assert _accepts(schema_parser(source), {"claims": [], "questions": []})


def test_actual_parser_keeps_claim_without_future_question():
    pytest.importorskip("lmformatenforcer")
    source = {"text": "Notably, treatment reduced IL6 in human cells."}
    value = payload()
    value["questions"] = []
    assert _accepts(schema_parser(source), value)
    value["questions"] = [{"quote": "Future studies should invent an endpoint."}]
    assert not _accepts(schema_parser(source), value)


def test_empty_array_guard_ignores_punctuation_in_nested_strings():
    pytest.importorskip("lmformatenforcer")
    source = {"text": "Notably, treatment reduced IL6 in human cells."}
    value = payload()
    value["questions"] = []
    value["claims"][0]["object"] = 'IL6 {"questions": [ with \\ escaped punctuation'
    assert _accepts(schema_parser(source), value)


def _character_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocabulary = {"<unk>": 0, "<eos>": 1}
    vocabulary.update({char: index + 2 for index, char in enumerate(string.printable)})
    tokenizer_impl = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer_impl.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tokenizer_impl.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_impl, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>"
    )


def test_actual_prefix_hook_forbids_nonempty_question_after_array_opener():
    pytest.importorskip("lmformatenforcer")
    torch = pytest.importorskip("torch")
    tokenizer = _character_tokenizer()
    prefix, _ = prefix_constraint(tokenizer, {"text": "Treatment reduced IL6 in human cells."})
    sequence = tokenizer.encode("0")
    for char in '{"claims": [], "questions": [':
        allowed = prefix(0, torch.tensor(sequence))
        token = tokenizer.encode(char)[0]
        assert token in allowed
        sequence.append(token)
    allowed = prefix(0, torch.tensor(sequence))
    assert tokenizer.encode("]")[0] in allowed
    assert tokenizer.encode("{")[0] not in allowed
    assert tokenizer.encode('"')[0] not in allowed


def test_actual_tiny_cpu_generation_obeys_empty_source_schema():
    """Real Transformers generate integration with random tiny CPU weights only."""
    pytest.importorskip("lmformatenforcer")
    torch = pytest.importorskip("torch")
    from transformers import GPT2Config, GPT2LMHeadModel

    tokenizer = _character_tokenizer()
    torch.manual_seed(1729)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=256,
            n_embd=16,
            n_layer=1,
            n_head=2,
            eos_token_id=1,
            pad_token_id=1,
        )
    ).eval()
    source = {"text": "Incomplete source fragment"}
    prefix, _ = prefix_constraint(tokenizer, source)
    inputs = tokenizer("0", return_tensors="pt")
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            prefix_allowed_tokens_fn=prefix,
            pad_token_id=1,
        )
    decoded = tokenizer.decode(
        generated[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )
    assert json.loads(decoded) == {"claims": [], "questions": []}
