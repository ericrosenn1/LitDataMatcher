import json

from litdatamatcher.pipeline import run_pipeline


def test_pipeline_writes_reproducible_artifacts(tmp_path):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "title": "Microbiome recovery",
                "abstract": (
                    "Further research should examine whether antibiotic exposure and "
                    "longitudinal microbiome data explain recovery."
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = run_pipeline(input_path, tmp_path / "out", top_n=5)

    assert metrics["documents"] == 1
    assert metrics["questions"] >= 1
    assert metrics["matches"] >= 1
    assert (tmp_path / "out" / "questions.jsonl").exists()
    assert (tmp_path / "out" / "matches.jsonl").exists()
    assert (tmp_path / "out" / "litdatamatcher.sqlite").exists()
