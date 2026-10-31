"""Self-consistent synthetic receipts test provenance, not scientific validity."""

import copy
import json

import pytest

from scripts.campaign_20261031.audit_case_batch import audit, file_sha


class NativeFixture:
    def __init__(self, root):
        self.root = root
        self.batch = root / "batch"
        self.run = self.batch / "cases/case_1/attempt-0001"
        self.inputs = root / "inputs.jsonl"
        self.text = "Treatment reduced IL6. Further studies should assess β-cell durability."
        self.caller_text = "Does treatment reduce IL6?"
        self.input = {
            "case_id": "case_1",
            "document_id": "doc:1",
            "text": self.text,
            "question": self.caller_text,
        }
        self.questions = [
            {
                "question_id": "q_user",
                "origin": "user",
                "question": self.caller_text,
                "source_document_id": "doc:1",
            }
        ]
        self.claims = []
        self.matches = [{"question_id": "q_user", "dataset_id": "dataset:1"}]
        self.outcomes = [{"question_id": "q_user", "classification": "REQUIRES_INSPECTION"}]
        self.extraction = {"questions": [], "evidence": []}

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    @staticmethod
    def write_rows(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in value), encoding="utf-8"
        )

    def source_question(self, origin="explicit_unresolved_source"):
        text = "Further studies should assess β-cell durability."
        start = self.text.index(text)
        return {
            "question_id": "q_source",
            "origin": origin,
            "question": text,
            "text": text,
            "source_document_id": "doc:1",
            "evidence_span": {"start": start, "end": start + len(text), "text": text},
        }

    def seal(self):
        self.write_rows(self.inputs, [self.input])
        payloads = {
            "claims.jsonl": self.claims,
            "questions.jsonl": self.questions,
            "matches.jsonl": self.matches,
            "candidate_outcomes.jsonl": self.outcomes,
        }
        for name, value in payloads.items():
            self.write_rows(self.run / name, value)
        documents = [{"document_id": "doc:1", "text": self.text}]
        source = self.run / "local_ingestion/literature.jsonl"
        self.write_rows(source, documents)
        ingestion = {
            "records": documents,
            "outputs": [{"name": "literature.jsonl", "identity": {"sha256": file_sha(source)}}],
        }
        stages = {}
        for name, value in (("local_ingestion", ingestion), ("extraction", self.extraction)):
            path = self.run / ".workflow" / f"{name}.json"
            self.write_json(path, value)
            stages[name] = {"file": path.name, "sha256": file_sha(path)}
        workflow = self.run / ".workflow/state.json"
        self.write_json(
            workflow,
            {"status": "COMPLETE", "completed": {"result": {"status": "PASS"}}, "stages": stages},
        )
        manifest = self.run / "RUN_MANIFEST.json"
        self.write_json(
            manifest,
            {
                "execution_status": "PASS",
                "failures": [],
                "artifacts": [
                    {"path": name, "sha256": file_sha(self.run / name)} for name in payloads
                ],
            },
        )
        outputs = [self.run / name for name in payloads] + [manifest, workflow]
        self.write_json(
            self.batch / "BATCH_STATE.json",
            {
                "inputs": {"input": {"sha256": file_sha(self.inputs)}},
                "cases": {
                    "case_1": {
                        "terminal": True,
                        "run_dir": self.run.relative_to(self.batch).as_posix(),
                        "outputs": [
                            {"path": p.relative_to(self.batch).as_posix(), "sha256": file_sha(p)}
                            for p in outputs
                        ],
                    }
                },
            },
        )
        return self

    def audit(self):
        self.seal()
        return audit(self.batch, self.inputs, self.root / "audit.json")


@pytest.fixture
def fixture(tmp_path):
    return NativeFixture(tmp_path)


def test_exact_caller_question_and_known_output_links_pass(fixture):
    report = fixture.audit()
    assert report["status"] == "PASS_ARTIFACT_AND_SPAN_AUDIT"
    assert report["question_origins"] == {"user": 1}
    assert "no expert accuracy" in report["limitations"]


def test_no_frozen_caller_question_does_not_require_one(fixture):
    del fixture.input["question"]
    fixture.questions = fixture.matches = fixture.outcomes = []
    assert fixture.audit()["question_origins"] == {}


@pytest.mark.parametrize("tamper", [False, True])
def test_existing_literal_claim_audit_is_preserved(fixture, tamper):
    quote = "Treatment reduced IL6."
    fixture.claims = [
        {
            "source_document_id": "doc:1",
            "quote": quote,
            "evidence_span": {"start": 0, "end": len(quote), "text": quote},
            "subject": "Treatment",
            "predicate": "reduced",
            "object": "IL6",
        }
    ]
    if tamper:
        fixture.claims[0]["quote"] = "Treatment increased IL6."
        with pytest.raises(ValueError, match="Claim source span failure"):
            fixture.audit()
    else:
        assert fixture.audit()["accepted_claims"] == 1


@pytest.mark.parametrize("nested", [False, True])
def test_query_fallback_and_nested_source_envelope_pass(fixture, nested):
    fixture.input["query"] = fixture.input.pop("question")
    if nested:
        fixture.input["source_document"] = {
            "id": fixture.input.pop("document_id"),
            "text": fixture.input.pop("text"),
        }
    assert fixture.audit()["cases"] == 1


@pytest.mark.parametrize(
    "change", ["text", "source", "missing", "unexpected", "duplicate", "alias"]
)
def test_caller_provenance_mismatch_is_rejected_after_checksums_are_resealed(fixture, change):
    if change == "text":
        fixture.questions[0]["question"] += " "
    elif change == "source":
        fixture.questions[0]["source_document_id"] = "doc:other"
    elif change == "missing":
        fixture.questions = []
        fixture.matches = fixture.outcomes = []
    elif change == "unexpected":
        del fixture.input["question"]
    elif change == "duplicate":
        fixture.questions.append({**fixture.questions[0], "question_id": "q_second"})
    else:
        fixture.questions[0]["text"] = "A different displayed question."
    with pytest.raises(ValueError, match="Caller|aliases"):
        fixture.audit()


@pytest.mark.parametrize("origin", ["explicit_unresolved", "explicit_unresolved_source"])
def test_source_extracted_question_requires_exact_parent_text(fixture, origin):
    fixture.questions.append(fixture.source_question(origin))
    fixture.matches.append({"question_id": "q_source"})
    result = fixture.audit()
    assert result["question_origins"][origin] == 1


@pytest.mark.parametrize(
    "change", ["text", "span_text", "source", "negative", "too_long", "bool", "float", "missing"]
)
def test_source_question_span_tampering_is_rejected(fixture, change):
    question = fixture.source_question()
    if change == "text":
        question["question"] = question["text"] = "A different question?"
    elif change == "span_text":
        question["evidence_span"]["text"] = "changed"
    elif change == "source":
        question["source_document_id"] = "doc:other"
    elif change == "negative":
        question["evidence_span"]["start"] = -1
    elif change == "too_long":
        question["evidence_span"]["end"] = len(fixture.text) + 1
    elif change == "bool":
        question["evidence_span"]["start"] = True
    elif change == "float":
        question["evidence_span"]["start"] = 0.0
    else:
        del question["evidence_span"]
    fixture.questions.append(question)
    with pytest.raises(ValueError, match="[Qq]uestion"):
        fixture.audit()


@pytest.mark.parametrize("origin", ["model_guess", None, ["user"]])
def test_unknown_origins_are_rejected(fixture, origin):
    fixture.questions[0]["origin"] = origin
    with pytest.raises(ValueError, match="Unknown question origin"):
        fixture.audit()


def test_duplicate_question_identifiers_are_rejected(fixture):
    question = fixture.source_question()
    question["question_id"] = fixture.questions[0]["question_id"]
    fixture.questions.append(question)
    with pytest.raises(ValueError, match="Duplicate question ID"):
        fixture.audit()


@pytest.mark.parametrize("collection", ["matches", "outcomes"])
@pytest.mark.parametrize("identifier", ["q_missing", None, []])
def test_dangling_output_question_references_are_rejected(fixture, collection, identifier):
    getattr(fixture, collection)[0]["question_id"] = identifier
    with pytest.raises(ValueError, match="Unknown question reference"):
        fixture.audit()


def test_known_derived_origin_uses_recorded_evidence_links_without_semantic_judgment(fixture):
    question = {
        "question_id": "q_derived",
        "origin": "cross_document_conflict",
        "question": "What explains this recorded conflict?",
        "proposition_id": "prop:1",
        "source_evidence_ids": ["ev:1"],
    }
    fixture.questions.append(question)
    fixture.extraction = {
        "questions": [copy.deepcopy(question)],
        "evidence": [{"evidence_id": "ev:1"}],
    }
    assert fixture.audit()["question_origins"]["cross_document_conflict"] == 1


@pytest.mark.parametrize("change", ["record", "evidence"])
def test_derived_question_requires_known_record_and_evidence(fixture, change):
    question = {
        "question_id": "q_derived",
        "origin": "cross_document_conflict",
        "question": "What explains this recorded conflict?",
        "source_evidence_ids": ["ev:1"],
    }
    fixture.questions.append(question)
    fixture.extraction = {
        "questions": [copy.deepcopy(question)],
        "evidence": [{"evidence_id": "ev:1"}],
    }
    if change == "record":
        fixture.extraction["questions"][0]["question"] = "Changed recorded derivation?"
    else:
        fixture.extraction["evidence"] = []
    with pytest.raises(ValueError, match="Derived question"):
        fixture.audit()
