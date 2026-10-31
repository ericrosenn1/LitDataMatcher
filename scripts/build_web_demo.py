"""Build the browser demo from the existing deterministic Python pipeline.

The source paragraphs are authored illustrations, not published studies.
Catalog records describe source collections, not independently validated
accession-level datasets. No measured sample counts are asserted by the UI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from litdatamatcher.literature import analyze_literature_records, document_id  # noqa: E402
from litdatamatcher.meta_analysis import run_meta_analysis_node, synthesis_index  # noqa: E402
from litdatamatcher.pipeline import collect_candidate_datasets  # noqa: E402
from litdatamatcher.ranking import rank_matches  # noqa: E402


RECORDS = [
    {
        "title": "Microbiome recovery after antibiotic exposure",
        "abstract": "Longitudinal sampling can characterize the recovery of human gut microbial communities after antibiotic exposure. Can longitudinal microbiome profiles predict recovery after antibiotic exposure?",
    },
    {
        "title": "Diet, microbial metabolism, and intestinal inflammation",
        "abstract": "Diet and microbial metabolism may contribute to variation in intestinal inflammation. Does dietary variation explain differences in microbial metabolite profiles during intestinal inflammation?",
    },
    {
        "title": "Host transcriptional response in inflammatory bowel disease",
        "abstract": "Human host gene expression offers a complementary view of inflammatory bowel disease and its clinical course. Do host transcriptomic signatures distinguish treatment responders from non-responders in inflammatory bowel disease?",
    },
    {
        "title": "Microbial predictors of clinical remission",
        "abstract": "Linking baseline microbiome composition to subsequent outcomes requires longitudinal human data. Are baseline microbial communities associated with subsequent clinical remission?",
    },
    {
        "title": "Following disease activity over time",
        "abstract": "Repeated measurements of clinical outcomes may help characterize the course of inflammatory bowel disease in human participants. Can repeated clinical measurements distinguish transient inflammation from persistent disease activity?",
    },
    {
        "title": "Connecting microbiome and metabolomics measurements",
        "abstract": "Multi-omic approaches may connect human gut microbial composition to metabolic readouts and clinical outcomes. Does combining metabolomics with microbiome composition improve prediction of treatment response?",
    },
]


def main() -> None:
    questions = analyze_literature_records(RECORDS)
    syntheses = run_meta_analysis_node(questions)
    datasets = collect_candidate_datasets(questions)
    matches = rank_matches(questions, datasets, synthesis_index(syntheses), top_n=100)
    weights = [
        ("Significance", "significance", 0.32),
        ("Feasibility heuristic", "feasibility", 0.28),
        ("Variable overlap", "variable_overlap", 0.16),
        ("Lexical relevance", "semantic_relevance", 0.10),
        ("Catalog quality heuristic", "data_quality", 0.08),
        ("Sample-size heuristic", "sample_adequacy", 0.06),
        ("Uncertainty penalty (subtracted)", "uncertainty_penalty", 0.18),
    ]
    workspace = {
        "id": "demo-microbiome-v1",
        "name": "The microbiome & human health",
        "mode": "demo",
        "createdAt": "2026-10-31T00:00:00Z",
        "papers": [
            {
                "id": document_id(record),
                "title": record["title"],
                "abstract": record["abstract"],
                "source": "Illustrative research brief",
                "isIllustrative": True,
            }
            for record in RECORDS
        ],
        "datasets": [
            {
                "id": dataset.dataset_id,
                "title": dataset.title,
                "description": dataset.description,
                "url": dataset.url,
                "source": dataset.source,
                "sampleSize": None,
                "variables": [variable.name for variable in dataset.variables],
                "population": dataset.populations,
                "access": dataset.access_type,
                "license": dataset.license,
                "isIllustrative": True,
            }
            for dataset in datasets
        ],
        "opportunities": [
            {
                "id": match.match_id,
                "question": match.question.question,
                "evidence": match.question.evidence[0].text,
                "extractionType": match.question.extraction_type,
                "paperId": match.question.source_ids[0],
                "datasetId": match.dataset.dataset_id,
                "score": round(match.score.combined * 100, 1),
                "components": [
                    {"label": label, "score": round(getattr(match.score, key) * 100, 1), "weight": weight}
                    for label, key, weight in weights
                ],
                "rationale": match.rationale,
                "caveats": [
                    "Illustrative source text and curated source-collection metadata; not a validated scientific finding.",
                    "Catalog sample counts, quality values, and completeness are demonstration assumptions used by the existing Python scorer.",
                    *([f"Missing required variables: {', '.join(match.missing_variables)}."] if match.missing_variables else []),
                    *match.assessments.get("feasibility", {}).get("caveats", []),
                ],
                "terms": match.question.domain_terms,
            }
            for match in matches
        ],
        "warnings": [
            "Demo workspace: six authored research briefs, five curated source collections, and real outputs from the deterministic Python scorer. These are illustrative, not published findings.",
            "Repository links are real; collection metadata, sample counts, quality, and completeness are demonstration assumptions. Validate individual accessions before research use.",
            "Ranking scores are explainable heuristics, not confidence probabilities or evidence of scientific validity.",
        ],
        "provenance": {
            "algorithm": "python-pipeline-demo",
            "generator": "scripts/build_web_demo.py",
            "sourceType": "authored illustrative research briefs + existing CuratedBiomedicalCatalogAdapter",
            "scoreFormula": "0.32 significance + 0.28 feasibility + 0.16 variable overlap + 0.10 lexical relevance + 0.08 catalog quality + 0.06 sample heuristic - 0.18 uncertainty penalty",
            "originalMatches": [match.to_dict() for match in matches],
            "inputRecords": RECORDS,
        },
    }
    target = ROOT / "web" / "src" / "data" / "demo.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(workspace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Built {target}: {len(RECORDS)} briefs, {len(datasets)} source collections, {len(matches)} matches.")


if __name__ == "__main__":
    main()
