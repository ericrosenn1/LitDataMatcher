"""Engineering-only local fixtures; these tests are not source-link quality evidence."""

import json
from dataclasses import replace

import pytest

from litdatamatcher.data_plane import atomic_json
from litdatamatcher.retrieval_evaluation import evaluate_retrieval
from litdatamatcher.retrieval_training import (
    FIXED_CONFIG,
    file_sha,
    promotion_decision,
    run_training,
    validate_benchmark,
)


def fixture_benchmark(root, train_count=16, development_count=4):
    root.mkdir()
    rows, documents = {"train": [], "development": []}, []
    for index in range(train_count + development_count):
        split = "train" if index < train_count else "development"
        identity = f"pmid:{index + 1}"
        documents.append(
            {
                "id": identity,
                "text": f"study token{index} human response\u2028source {index}",
                "pmid": str(index + 1),
            }
        )
        rows[split].append(
            {
                "id": f"q{index}",
                "query": f"does token{index} predict response {index}",
                "positive_doc_id": identity,
                "source_corpus": "synthetic_fixture",
                "split": split,
                "provenance_relative_path": "synthetic/fixture.json",
            }
        )
    for name, value in {**rows, "documents": documents}.items():
        (root / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in value), encoding="utf-8"
        )
    update_manifest(root)
    return root


def update_manifest(root):
    paths = [root / f"{name}.jsonl" for name in ("train", "development", "documents")]
    manifest = {
        "status": "PASS",
        "purpose": "Synthetic engineering fixture, not scientific evidence",
        "counts": {p.stem: len(p.read_text(encoding="utf-8").split("\n")) - 1 for p in paths},
        "leakage_checks": {
            key: 0
            for key in (
                "canonical_paper_overlap",
                "canonical_query_overlap",
                "canonical_document_text_overlap",
            )
        },
        "outputs": [
            {"path": p.name, "sha256": file_sha(p), "bytes": p.stat().st_size} for p in paths
        ],
    }
    atomic_json(root / "BENCHMARK_MANIFEST.json", manifest)


def change_rows(root, name, function):
    path = root / f"{name}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]
    function(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    update_manifest(root)


def test_benchmark_receipts_and_unicode_source_text(tmp_path):
    root = fixture_benchmark(tmp_path / "benchmark")
    result = validate_benchmark(root)
    assert len(result["train"]) == 16
    assert "\u2028" in result["documents"][0]["text"]
    with (root / "train.jsonl").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="integrity failure"):
        validate_benchmark(root)


@pytest.mark.parametrize(
    "kind",
    ["query", "paper", "alias", "text", "duplicate_train", "unknown_positive", "wrong_split"],
)
def test_independent_leakage_and_contract_rejection(tmp_path, kind):
    root = fixture_benchmark(tmp_path / "benchmark")
    if kind == "query":
        change_rows(
            root,
            "development",
            lambda rows: rows[0].update(query="DOES  token0 predict response 0"),
        )
    elif kind == "paper":
        change_rows(root, "development", lambda rows: rows[0].update(positive_doc_id="pmid:1"))
    elif kind == "alias":
        change_rows(root, "documents", lambda rows: rows[16].update(pmid="1"))
    elif kind == "text":
        change_rows(root, "documents", lambda rows: rows[16].update(text=rows[0]["text"]))
    elif kind == "duplicate_train":
        change_rows(
            root, "train", lambda rows: rows[1].update(positive_doc_id=rows[0]["positive_doc_id"])
        )
    elif kind == "unknown_positive":
        change_rows(root, "train", lambda rows: rows[0].update(positive_doc_id="missing"))
    else:
        change_rows(root, "development", lambda rows: rows[0].update(split="evaluation_candidate"))
    with pytest.raises(ValueError):
        validate_benchmark(root)


def test_promotion_retains_incumbent_unless_predeclared_thresholds_pass():
    queries = [{"id": str(i), "positive_doc_id": str(i)} for i in range(4)]
    baseline = {
        "metrics": evaluate_retrieval(
            queries, [[{"id": "other"}, {"id": str(i)}] for i in range(4)]
        )
    }
    same = {"metrics": baseline["metrics"]}
    better = {"metrics": evaluate_retrieval(queries, [[{"id": str(i)}] for i in range(4)])}
    assert promotion_decision(baseline, {"epoch-01": same})["selected"] == "pretrained"
    result = promotion_decision(baseline, {"epoch-01": better, "epoch-02": better})
    assert result["selected"] == "epoch-01"
    assert result["status"] == "EXPERIMENTAL_DEVELOPMENT_SELECTION"


def tiny_local_model(path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    path.mkdir()
    vocabulary = (
        [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "study",
            "human",
            "response",
            "source",
            "does",
            "predict",
        ]
        + [f"token{i}" for i in range(40)]
        + [str(i) for i in range(40)]
    )
    (path / "vocab.txt").write_text("\n".join(vocabulary))
    tokenizer = transformers.BertTokenizerFast(
        vocab_file=str(path / "vocab.txt"), do_lower_case=True
    )
    tokenizer.save_pretrained(path)
    torch.manual_seed(1729)
    config = transformers.BertConfig(
        vocab_size=len(vocabulary),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=512,
    )
    transformers.BertModel(config).save_pretrained(path, safe_serialization=True)
    files = [
        {"path": p.name, "sha256": file_sha(p), "bytes": p.stat().st_size}
        for p in sorted(path.iterdir())
    ]
    atomic_json(
        path / "MODEL_MANIFEST.json",
        {
            "model_id": "local/synthetic-test-fixture",
            "revision": "1" * 40,
            "revision_type": "synthetic_test_fixture",
            "license": "synthetic-test-only",
            "files": files,
        },
    )
    return path


def test_actual_tiny_cpu_training_resume_and_safe_epoch_artifacts(tmp_path, monkeypatch):
    import litdatamatcher.retrieval_training as trainer

    benchmark = fixture_benchmark(tmp_path / "benchmark")
    model = tiny_local_model(tmp_path / "model")
    original = file_sha(model / "model.safetensors")
    interrupted = tmp_path / "interrupted"

    def stop_after_first(event):
        if event["stage"] == "epoch_saved" and event["epoch"] == 1:
            raise KeyboardInterrupt("synthetic interruption after valid first epoch")

    with pytest.raises(KeyboardInterrupt):
        run_training(benchmark, model, interrupted, device="cpu", progress=stop_after_first)
    assert (interrupted / "epoch-01/EPOCH_RECEIPT.json").is_file()
    assert len(list(interrupted.glob("FAILURE_*.json"))) == 1
    assert json.loads((interrupted / "TRAINING_STATE.json").read_text())["status"] == "INTERRUPTED"
    first_weight_hash = file_sha(interrupted / "epoch-01/model.safetensors")
    assert first_weight_hash != original
    result = run_training(benchmark, model, interrupted, device="cpu", resume=True)
    assert result["status"] == "PASS"
    assert file_sha(interrupted / "epoch-01/model.safetensors") == first_weight_hash
    for epoch in (1, 2):
        manifest = json.loads((interrupted / f"epoch-{epoch:02d}/MODEL_MANIFEST.json").read_text())
        assert len(manifest["revision"]) == 40
        assert manifest["revision_type"] == "local_training_content_fingerprint"
        assert manifest["parent_model"]["revision"]
        assert manifest["training_provenance"]["runtime_versions"]["torch"]
        assert manifest["training_provenance"]["benchmark_manifest_sha256"]
    baseline = tmp_path / "uninterrupted"
    run_training(benchmark, model, baseline, device="cpu")
    assert file_sha(interrupted / "epoch-02/model.safetensors") == file_sha(
        baseline / "epoch-02/model.safetensors"
    )
    assert file_sha(model / "model.safetensors") == original

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed epochs must not train again")

    monkeypatch.setattr(trainer, "train_epoch", forbidden)
    replay = run_training(benchmark, model, interrupted, device="cpu", resume=True)
    assert replay["selection"] == result["selection"]
    assert "error" not in json.loads((interrupted / "TRAINING_STATE.json").read_text())
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run_training(benchmark, model, interrupted, device="cuda", resume=True)
    with (interrupted / "epoch-01/optimizer.pt").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="integrity failure"):
        run_training(benchmark, model, interrupted, device="cpu", resume=True)


def test_fixed_configuration_is_not_silently_changed(tmp_path):
    with pytest.raises(ValueError, match="fixed training configuration"):
        run_training(tmp_path, tmp_path, tmp_path, config=replace(FIXED_CONFIG, batch_size=8))


def test_evaluation_cache_reuses_only_integrity_checked_results(tmp_path):
    from litdatamatcher.retrieval_training import _cached_evaluation

    destination = tmp_path / "evaluation.json"
    calls = []
    assert _cached_evaluation(destination, "fixed", lambda: calls.append(1) or {"result": 1}) == {
        "result": 1
    }
    assert _cached_evaluation(destination, "fixed", lambda: calls.append(2)) == {"result": 1}
    assert calls == [1]
    destination.write_text('{"result":2}')
    with pytest.raises(ValueError, match="integrity failure"):
        _cached_evaluation(destination, "fixed", lambda: calls.append(3))
    assert calls == [1]
