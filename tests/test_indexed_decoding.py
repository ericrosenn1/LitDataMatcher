"""Synthetic source-index transport checks; no real-case accuracy labels."""

import copy
import importlib.metadata
import json
import string

import pytest

from litdatamatcher import indexed_decoding as indexed
from litdatamatcher import semantic_runtime
from litdatamatcher.semantic_runtime import LocalSemanticRuntime, RuntimeConfig, validate_extraction
from litdatamatcher.structured_decoding import schema_parser as quote_parser


def document():
    return {
        "document_id": "synthetic:indexed",
        "title": "Synthetic quoted groups",
        "text": 'Treatment reduced IL6 in "group A" cells. Future studies should assess β-cell durability.',
    }


def payload(index=0):
    return {
        "claims": [
            {
                "sentence_index": index,
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
        "questions": [{"sentence_index": 1}],
    }


def accepts(parser, value):
    for char in json.dumps(value, ensure_ascii=False, separators=(",", ":")):
        if char not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(char)
    return parser.can_end()


@pytest.fixture
def fake_runtime(monkeypatch):
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "0.11.3" if name == "lm-format-enforcer" else "synthetic-version",
    )
    runtime = LocalSemanticRuntime.__new__(LocalSemanticRuntime)
    runtime.config = RuntimeConfig(decoding_mode="source_index")
    runtime.model_manifest = {"model_id": "synthetic", "revision": "a" * 40, "license": "fixture"}
    runtime._generate = lambda *args: (json.dumps(payload()), 100, 50)
    return runtime


def test_quoted_sentence_is_looked_up_exactly_before_unchanged_guard():
    source = document()
    wire = payload()
    mapped, trace = indexed.map_index_payload(wire, source)
    assert mapped["claims"][0]["quote"] == 'Treatment reduced IL6 in "group A" cells.'
    assert wire == payload()  # Mapping does not mutate raw transport evidence.
    assert "sentence_index" not in mapped["claims"][0]
    result = validate_extraction(mapped, source)
    assert len(result["claims"]) == len(result["questions"]) == 1
    assert trace["selected_sentences"][0]["start"] == 0


def test_actual_index_grammar_avoids_quoted_string_enum_failure():
    pytest.importorskip("lmformatenforcer")
    mapped, _ = indexed.map_index_payload(payload(), document())
    assert not accepts(quote_parser(document()), mapped)
    assert accepts(indexed.schema_parser(document()), payload())


@pytest.mark.parametrize("index", [True, False, 0.0, "0", None, -1, 2, 2**63])
def test_invalid_index_types_and_ranges_are_rejected(index):
    with pytest.raises(ValueError):
        indexed.map_index_payload(payload(index), document())


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_field",
        "quote_field",
        "missing_index",
        "observed_question",
        "duplicate_question",
        "too_many_claims",
    ],
)
def test_wire_contract_and_future_membership_are_strict(mutation):
    wire = payload()
    if mutation == "unknown_field":
        wire["claims"][0]["score"] = 1
    elif mutation == "quote_field":
        wire["claims"][0]["quote"] = "invented"
    elif mutation == "missing_index":
        del wire["claims"][0]["sentence_index"]
    elif mutation == "observed_question":
        wire["questions"][0]["sentence_index"] = 0
    elif mutation == "duplicate_question":
        wire["questions"].append(copy.deepcopy(wire["questions"][0]))
    else:
        wire["claims"].append(copy.deepcopy(wire["claims"][0]))
    with pytest.raises(ValueError):
        indexed.map_index_payload(wire, document())


@pytest.mark.parametrize(
    "raw", ['{"claims":[],"claims":[],"questions":[]}', '{"claims":[],"questions":[],"x":NaN}']
)
def test_raw_json_duplicate_keys_and_nonfinite_numbers_are_rejected(raw):
    with pytest.raises(ValueError):
        indexed.parse_index_json(raw)


def test_source_table_preserves_whitespace_unicode_and_abbreviations():
    source = {
        "text": "Dr. Smith reported results. Treatment reduced IL6 in cafe\u0301 cells.\r\nFuture studies should assess β-cell durability. Trailing fragment"
    }
    entries = indexed.sentence_table(source)
    assert len(entries) == 3
    assert entries[0]["text"] == "Dr. Smith reported results."
    assert "cafe\u0301" in entries[1]["text"]
    for entry in entries:
        assert source["text"][entry["start"] : entry["end"]] == entry["text"]
    repeated = {
        "text": "Treatment reduced IL6. Treatment reduced IL6. Future studies should assess safety."
    }
    assert [row["text"] for row in indexed.sentence_table(repeated)] == [
        "Future studies should assess safety."
    ]


def test_wrong_direction_and_roles_from_another_sentence_still_fail_guard():
    for change in ({"direction": "increase"}, {"subject": "Future studies"}):
        wire = payload()
        wire["claims"][0].update(change)
        mapped, _ = indexed.map_index_payload(wire, document())
        result = validate_extraction(mapped, document())
        assert not result["claims"] and result["rejected"]


def test_future_sentence_selected_as_claim_is_not_promoted():
    wire = payload(1)
    wire["claims"][0].update(
        subject="Future studies",
        predicate="should assess",
        object="β-cell durability",
        direction="unknown",
    )
    mapped, _ = indexed.map_index_payload(wire, document())
    assert not validate_extraction(mapped, document())["claims"]


def test_map_fingerprint_tracks_order_text_and_offsets_but_backend_is_static(fake_runtime):
    source = document()
    first = fake_runtime.inference_fingerprint(source)
    for text in (
        'Future studies should assess β-cell durability. Treatment reduced IL6 in "group A" cells.',
        source["text"].replace("group A", "group B"),
        " " + source["text"],
    ):
        second = fake_runtime.inference_fingerprint({**source, "text": text})
        assert first["decoding"]["sentence_map_sha256"] != second["decoding"]["sentence_map_sha256"]
    backend = fake_runtime.backend_identity()
    assert "sentence_map_sha256" not in backend["decoding"]
    assert backend["prompt_version"] == indexed.PROMPT_VERSION


def test_raw_indices_mapping_and_public_quotes_survive_cache_replay(fake_runtime, tmp_path):
    source = document()
    result = fake_runtime.extract(source, tmp_path)
    trace = result["decoding_trace"]
    assert trace["attempts"][0]["raw_payload"] == payload()
    assert (
        trace["sentence_map_sha256"]
        == result["inference_manifest"]["decoding"]["sentence_map_sha256"]
    )
    assert result["claims"][0]["quote"] == trace["sentence_table"][0]["text"]
    assert list((tmp_path / "diagnostics").rglob("*.txt"))
    fake_runtime._generate = lambda *args: pytest.fail("Cache must replay")
    replay = fake_runtime.extract(source, tmp_path)
    assert replay["decoding_trace"] == trace
    assert replay["inference_manifest"]["origin"] == "cache_replay"


def test_original_source_mutation_does_not_change_bound_snapshot(fake_runtime):
    source = document()
    original_text = source["text"]

    def generate(*args):
        source["text"] = "A different source was inserted."
        return json.dumps(payload()), 100, 50

    fake_runtime._generate = generate
    result = fake_runtime.extract(source)
    assert result["decoding_trace"]["sentence_table"][0]["text"] in original_text
    assert result["claims"][0]["quote"] in original_text


@pytest.mark.parametrize("index", [0, 1, 10, 127])
def test_actual_numeric_enum_accepts_full_valid_indices(index):
    pytest.importorskip("lmformatenforcer")
    source = {"text": " ".join(f"Treatment reduced IL6 for cohort {n}." for n in range(128))}
    wire = payload(index)
    wire["questions"] = []
    assert accepts(indexed.schema_parser(source), wire)
    mapped, _ = indexed.map_index_payload(wire, source)
    assert mapped["claims"][0]["quote"].endswith(f"cohort {index}.")


@pytest.mark.parametrize("index", [0.0, -1, 128, "1", None])
def test_actual_numeric_enum_forbids_invalid_grammar_indices(index):
    pytest.importorskip("lmformatenforcer")
    source = {"text": " ".join(f"Treatment reduced IL6 for cohort {n}." for n in range(128))}
    wire = payload(index)
    wire["questions"] = []
    assert not accepts(indexed.schema_parser(source), wire)


@pytest.mark.parametrize("text", ["Incomplete source fragment", "Treatment reduced IL6."])
def test_required_empty_arrays_remain_enforced_by_actual_parser(text):
    pytest.importorskip("lmformatenforcer")
    source = {"text": text}
    assert accepts(indexed.schema_parser(source), {"claims": [], "questions": []})
    assert not accepts(
        indexed.schema_parser(source), {"claims": [], "questions": [{"sentence_index": 0}]}
    )
    if "fragment" in text:
        wire = payload()
        wire["questions"] = []
        assert not accepts(indexed.schema_parser(source), wire)


def test_final_claim_cap_preserves_first_accepted_and_audits_overflow(fake_runtime, monkeypatch):
    source = {
        "document_id": "synthetic:repair",
        "text": "Treatment reduced IL6. Therapy reduced TNF.",
    }
    first, second = payload(), payload(1)
    first["questions"] = second["questions"] = []
    second["claims"][0].update(subject="Therapy", object="TNF")
    outputs = iter([first, second])
    fake_runtime._generate = lambda *args: (json.dumps(next(outputs)), 100, 50)
    guard = semantic_runtime.validate_extraction
    calls = []

    def retry_once(value, document):
        checked = guard(value, document)
        if not calls:
            checked["rejected"].append({"kind": "fixture", "reason": "Injected engineering retry"})
        calls.append(True)
        return checked

    monkeypatch.setattr(semantic_runtime, "validate_extraction", retry_once)
    result = fake_runtime.extract(source)
    assert len(result["claims"]) == 1
    assert result["claims"][0]["subject"] == "Treatment"
    overflow = result["decoding_trace"]["attempts"][1]["validated_overflow"]
    assert overflow[0]["record"]["subject"] == "Therapy"
    assert len(result["decoding_trace"]["attempts"]) == 2


def test_final_question_cap_preserves_first_three_validated_questions(fake_runtime):
    source = {
        "document_id": "synthetic:question-limit",
        "text": "Treatment reduced IL6. "
        + " ".join(f"Future studies should assess outcome {n}." for n in range(6)),
    }
    first, second = payload(), payload()
    first["claims"][0]["direction"] = "increase"  # Real guard rejection triggers repair.
    first["questions"] = [{"sentence_index": n} for n in (1, 2, 3)]
    second["questions"] = [{"sentence_index": n} for n in (4, 5, 6)]
    outputs = iter([first, second])
    fake_runtime._generate = lambda *args: (json.dumps(next(outputs)), 100, 50)
    result = fake_runtime.extract(source)
    assert len(result["claims"]) == 1 and len(result["questions"]) == 3
    assert [row["text"] for row in result["questions"]] == [
        f"Future studies should assess outcome {n}." for n in range(3)
    ]
    assert len(result["decoding_trace"]["attempts"][1]["validated_overflow"]) == 3


def _character_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocabulary = {"<unk>": 0, "<eos>": 1}
    vocabulary.update({char: index + 2 for index, char in enumerate(string.printable)})
    implementation = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    implementation.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    implementation.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(
        tokenizer_object=implementation, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>"
    )


def test_actual_token_prefix_blocks_decimal_scientific_and_out_of_range_index():
    pytest.importorskip("lmformatenforcer")
    torch = pytest.importorskip("torch")
    tokenizer = _character_tokenizer()
    source = {"text": " ".join(f"Treatment reduced IL6 for cohort {n}." for n in range(128))}
    prefix, _ = indexed.prefix_constraint(tokenizer, source)
    sequence = tokenizer.encode("0")
    for char in '{"claims":[{"sentence_index":12':
        token = tokenizer.encode(char)[0]
        assert token in prefix(0, torch.tensor(sequence))
        sequence.append(token)
    allowed = prefix(0, torch.tensor(sequence))
    assert tokenizer.encode("7")[0] in allowed
    for char in "89.e-":
        assert tokenizer.encode(char)[0] not in allowed


def test_actual_tiny_cpu_generation_of_index_payload():
    """Random tiny weights plus explicit token preferences test transport, not learned quality."""
    pytest.importorskip("lmformatenforcer")
    torch = pytest.importorskip("torch")
    from transformers import GPT2Config, GPT2LMHeadModel, LogitsProcessor

    tokenizer = _character_tokenizer()
    target = tokenizer.encode(json.dumps(payload(), separators=(",", ":")))

    class PreferFixture(LogitsProcessor):
        def __call__(self, input_ids, scores):
            position = input_ids.shape[-1] - 1
            token = target[position] if position < len(target) else tokenizer.eos_token_id
            scores[:, token] += 1000  # Adding cannot resurrect grammar-masked -inf scores.
            return scores

    torch.manual_seed(1729)
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
    prefix, _ = indexed.prefix_constraint(tokenizer, document())
    inputs = tokenizer("0", return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=len(target) + 1,
            do_sample=False,
            prefix_allowed_tokens_fn=prefix,
            logits_processor=[PreferFixture()],
            pad_token_id=1,
        )
    value = indexed.parse_index_json(tokenizer.decode(output[0, 1:], skip_special_tokens=True))
    assert value == payload()
    mapped, _ = indexed.map_index_payload(value, document())
    assert len(validate_extraction(mapped, document())["claims"]) == 1


def test_index_qualification_matches_static_backend_and_keeps_dynamic_map(fake_runtime, tmp_path):
    from dataclasses import asdict

    from litdatamatcher.v2 import qualified_runtime

    fresh = fake_runtime.extract(document())
    fingerprint = fresh["inference_manifest"]["fingerprint"]
    report = {
        "qualification_schema": "runtime-qualification-v2.3",
        "status": "PASS",
        "backend_identity": fake_runtime.backend_identity(),
        "expected_identity": fingerprint,
        "configuration": asdict(fake_runtime.config),
        "replay_identity": copy.deepcopy(fingerprint),
        "replay_configuration": asdict(fake_runtime.config),
        "replay_origin": "cache_replay",
        "network_control": {"blocked_probe": ["synthetic-network-probe"]},
        "fresh": fresh,
    }
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert qualified_runtime(path, fake_runtime)[0]
    report["backend_identity"]["decoding"]["mapping_version"] = "changed-mapping"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert not qualified_runtime(path, fake_runtime)[0]


def test_invalid_raw_indices_are_preserved_without_fallback_quotes(fake_runtime, tmp_path):
    wire = payload(-1)
    fake_runtime._generate = lambda *args: (json.dumps(wire), 100, 50)
    with pytest.raises(ValueError, match="bounded attempts"):
        fake_runtime.extract(document(), tmp_path)
    diagnostics = list((tmp_path / "diagnostics").rglob("*.txt"))
    assert len(diagnostics) == fake_runtime.config.max_attempts
    assert all(
        json.loads(path.read_text())["claims"][0]["sentence_index"] == -1 for path in diagnostics
    )
    assert not list(tmp_path.glob("*.json"))


def test_previous_accepted_id_is_never_overwritten(fake_runtime, monkeypatch):
    first, second = payload(), payload()
    second["claims"][0]["status"] = "interpretation"
    outputs = iter([first, second])
    fake_runtime._generate = lambda *args: (json.dumps(next(outputs)), 100, 50)
    guard = semantic_runtime.validate_extraction
    calls = []

    def retry_once(value, document):
        checked = guard(value, document)
        if not calls:
            checked["rejected"].append({"kind": "fixture", "reason": "Injected engineering retry"})
        calls.append(True)
        return checked

    monkeypatch.setattr(semantic_runtime, "validate_extraction", retry_once)
    result = fake_runtime.extract(document())
    assert result["claims"][0]["status"] == "direct_experiment"
    assert len(result["claims"]) == 1


def test_private_source_mutation_fails_closed_before_mapping(fake_runtime):
    def generate(source, *args):
        source["text"] = "A changed source must not reuse old indices."
        return json.dumps(payload()), 100, 50

    fake_runtime._generate = generate
    with pytest.raises(RuntimeError, match="Source snapshot changed"):
        fake_runtime.extract(document())


def test_runtime_generation_uses_index_prompt_and_table_without_old_quote_instructions(monkeypatch):
    pytest.importorskip("lmformatenforcer")
    torch = pytest.importorskip("torch")
    tokenizer = _character_tokenizer()
    captured = {}

    def template(messages, **kwargs):
        captured["messages"] = messages
        return "0"

    monkeypatch.setattr(tokenizer, "apply_chat_template", template)

    class GenerationStub:
        def generate(self, **kwargs):
            captured["options"] = kwargs
            continuation = tokenizer.encode(json.dumps(payload()))
            return torch.tensor([[*kwargs["input_ids"][0].tolist(), *continuation]])

    runtime = LocalSemanticRuntime.__new__(LocalSemanticRuntime)
    runtime.config = RuntimeConfig(decoding_mode="source_index")
    runtime._tokenizer = tokenizer
    runtime._model = GenerationStub()
    raw, _, _ = runtime._generate(document())
    assert json.loads(raw) == payload()
    assert captured["messages"][0]["content"] == indexed.SYSTEM_PROMPT
    assert "copy it EXACTLY into quote" not in captured["messages"][0]["content"]
    source = json.loads(captured["messages"][1]["content"].split("\n", 1)[1])
    assert "source_sentences" in source and "text" not in source
    assert source["source_sentences"][0]["text"] == indexed.sentence_table(document())[0]["text"]
    assert callable(captured["options"]["prefix_allowed_tokens_fn"])
