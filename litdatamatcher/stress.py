"""Deterministic corpus stress helpers for pipeline smoke testing."""

from __future__ import annotations

from pathlib import Path

from .pipeline import run_pipeline
from .schemas import JsonDict, stable_id
from .storage import write_jsonl


STRESS_TOPICS = (
    "antibiotic exposure and longitudinal microbiome recovery",
    "metabolomics predictors of inflammatory bowel disease remission",
    "host transcriptomics and treatment response",
    "dietary exposure and gut microbial functional profiles",
    "clinical outcomes after microbiome-directed intervention",
)


def synthetic_literature_records(documents: int = 25) -> list[JsonDict]:
    """Create deterministic literature records for scalable smoke tests."""

    records: list[JsonDict] = []
    for index in range(max(0, int(documents))):
        topic = STRESS_TOPICS[index % len(STRESS_TOPICS)]
        title = f"Stress Corpus Paper {index + 1}: {topic.title()}"
        question = (
            f"Future studies should examine whether {topic} can be explained "
            "using public biomedical datasets with harmonized outcomes."
        )
        records.append(
            {
                "source_id": stable_id("source", "stress", index),
                "document_id": stable_id("doc", "stress", index),
                "title": title,
                "abstract": question,
                "text": f"{question} Methods and limitations are summarized for reproducible stress testing.",
                "doi": "",
            }
        )
    return records


def run_synthetic_stress_test(
    out_dir: str | Path,
    documents: int = 25,
    top_n: int = 100,
) -> JsonDict:
    """Write a synthetic corpus and run the canonical pipeline on it."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    literature_path = out_dir / "synthetic_literature.jsonl"
    records = synthetic_literature_records(documents)
    write_jsonl(literature_path, records)
    metrics = run_pipeline(literature_path, out_dir / "run", top_n=top_n)
    return {
        "documents_requested": documents,
        "literature": str(literature_path),
        **metrics,
    }
