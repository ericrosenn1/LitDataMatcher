# LitDataMatcher

LitDataMatcher is a reproducible research-automation pipeline for turning
scientific literature into ranked, reviewable question-data opportunities.

The motivating problem is simple: papers often describe unresolved questions,
limitations, future directions, and under-tested hypotheses, while public
repositories contain datasets that may already be able to address some of those
questions. LitDataMatcher connects those two worlds by extracting open research
questions from text, normalizing their data requirements, discovering relevant
public datasets, and ranking the most plausible question-dataset pairs for
expert review.

The current implementation is a functional research-software foundation. It is
offline-reproducible by default, produces auditable JSONL/SQLite artifacts, and
is organized into explicit scientific nodes that can be independently tested,
benchmarked, replaced, or extended.

## What It Does Today

LitDataMatcher currently supports:

- Literature-driven open-question extraction from JSONL records containing
  titles, abstracts, full text, and DOI/source metadata.
- Future-direction and limitation-derived question generation.
- Lightweight meta-analysis style clustering and evidence synthesis across
  related questions.
- Ontology-backed variable harmonization for biomedical concepts such as
  antibiotic exposure, microbiome composition, transcriptomics, metabolomics,
  disease activity, treatment, outcomes, and timepoints.
- Dataset discovery through a curated offline biomedical catalog.
- Dataset metadata normalization into stable `DatasetRecord` objects.
- Governance and reuse-risk scoring for access, license, and human-subject
  concerns.
- Pair-level feasibility assessment, including variable coverage, population
  fit, sample adequacy, assay fit, longitudinal fit, caveats, and recommended
  statistical design.
- Explainable ranking of question-dataset pairs.
- Expert review exports as CSV and JSONL.
- Evaluation utilities for question-extraction and ranking benchmarks.
- Manuscript-style Markdown reports for run summaries.
- A legacy streaming orchestrator for live topic-level demos.

## Scientific Workflow

The package pipeline follows this conceptual flow:

```text
literature records
  -> open-question identification
  -> meta-analysis style clustering
  -> dataset discovery and classification
  -> governance and feasibility assessment
  -> question-to-data ranking
  -> JSONL, SQLite, expert review sheets, and publication report
```

Each node writes inspectable outputs. This is intentional: the system is meant
to support scientific audit, error analysis, and expert-in-the-loop validation,
not just produce opaque scores.

## Repository Status

The repository has two execution paths:

- `litdatamatcher` package: the recommended reproducible pipeline for research
  and publication-oriented analyses.
- Top-level worker scripts: streaming subprocess entrypoints used by
  `orchestrator.py` for topic-level demonstrations and resource-management
  experiments.

The package path should be used for serious analyses. The orchestrator path is
kept for live demos and future distributed-worker development.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional NLP/model dependencies can be installed with:

```bash
python -m pip install -e ".[dev,nlp,ml]"
```

The deterministic core pipeline does not require GPU access, external APIs, or
network access.

## Quickstart

Run the built-in demo:

```bash
litdatamatcher demo --out run/demo
```

Equivalent module invocation:

```bash
python -m litdatamatcher.cli demo --out run/demo
```

Run on a JSONL literature corpus:

```bash
litdatamatcher run --input test_input.jsonl --out run/full --top-n 100
```

Each input row should include some combination of:

```json
{"title": "...", "abstract": "...", "text": "...", "doi": "..."}
```

## Outputs

A full run writes:

- `questions.jsonl`: extracted and normalized open research questions.
- `datasets.jsonl`: normalized candidate dataset records.
- `syntheses.jsonl`: question-cluster evidence summaries.
- `matches.jsonl`: ranked question-dataset opportunities with component scores.
- `summary.md`: concise human-readable ranking summary.
- `publication_report.md`: manuscript-style methods/results summary.
- `review_sheet.csv`: expert review sheet for relevance annotation.
- `review_sheet.jsonl`: programmatic review export.
- `litdatamatcher.sqlite`: queryable run database.
- `metrics.jsonl`: run-level counts and accounting.

The most important artifact for scientific review is usually `matches.jsonl` or
`review_sheet.csv`. Each match includes the question, dataset, component scores,
rationale, missing variables, feasibility assessment, governance assessment, and
recommended study design.

## Evaluation And Review

Evaluate a completed run against gold question labels:

```bash
litdatamatcher evaluate --run-dir run/full --gold-questions gold_questions.jsonl
```

Evaluate both extraction and ranking:

```bash
litdatamatcher evaluate ^
  --run-dir run/full ^
  --gold-questions gold_questions.jsonl ^
  --gold-ranking gold_ranking.jsonl ^
  --out run/full/evaluation.jsonl
```

Export a review sheet:

```bash
litdatamatcher review-export --run-dir run/full --out run/full/review.csv
```

Summarize completed review labels:

```bash
litdatamatcher review-summary --labels run/full/review.csv
```

Generate a manuscript-style report:

```bash
litdatamatcher report --run-dir run/full --out run/full/publication_report.md
```

## Streaming Orchestrator Demo

The legacy subprocess demo remains available:

```bash
python orchestrator.py --out run/orchestrator_matches.jsonl
```

Provide custom topics with:

```bash
python orchestrator.py --topics-file topics.txt --out run/orchestrator_matches.jsonl
```

`topics.txt` may be plain text, one topic per line, or JSONL with a `topic`,
`question`, or `title` field.

## Architecture

The main package modules are:

- `litdatamatcher.schemas`: validated dataclasses and stable IDs.
- `litdatamatcher.text`: deterministic text normalization, sentence splitting,
  sectioning, variable inference, and lexical similarity.
- `litdatamatcher.literature`: open-question extraction and significance
  scoring.
- `litdatamatcher.meta_analysis`: question clustering and evidence synthesis.
- `litdatamatcher.datasets`: dataset adapters, metadata normalization, and
  classification.
- `litdatamatcher.ontology`: variable harmonization and concept normalization.
- `litdatamatcher.governance`: access, license, privacy, and reuse-risk scoring.
- `litdatamatcher.feasibility`: pair-level design feasibility and caveat scoring.
- `litdatamatcher.ranking`: explainable question-to-dataset scoring.
- `litdatamatcher.storage`: JSONL and SQLite persistence.
- `litdatamatcher.evaluation`: extraction and ranking benchmark metrics.
- `litdatamatcher.review`: expert review exports and label summaries.
- `litdatamatcher.reporting`: publication-oriented run reports.
- `litdatamatcher.adapters`: optional live-source adapter scaffolds.
- `litdatamatcher.pipeline`: end-to-end orchestration.
- `litdatamatcher.cli`: command-line interface.

Additional documentation:

- `docs/architecture.md`
- `docs/node_contracts.md`
- `docs/reproducibility.md`
- `docs/model_robustness_plan.md`
- `docs/feature_roadmap.md`

## Testing

After installing development dependencies:

```bash
python -m pytest
python -m compileall litdatamatcher data_worker.py lit_gpu_worker.py matcher.py orchestrator.py
python -m litdatamatcher.cli demo --out run/demo
python -m litdatamatcher.cli report --run-dir run/demo
```

The deterministic tests are designed to run without GPU access, network access,
or external NLP models. Optional model-backed components should fail closed into
deterministic lexical fallbacks where possible.

## Reproducibility

LitDataMatcher is designed for auditable runs:

- Core identifiers are deterministic SHA-1 based stable IDs.
- Node outputs are JSONL and are also persisted to SQLite.
- Generated runtime files are ignored by `.gitignore`.
- Line-ending behavior is pinned through `.gitattributes`.
- Default dataset discovery is offline and deterministic.
- Optional HTTP adapters use caching and retry scaffolds.
- Scores are decomposed into interpretable components.

For a publication run, archive:

- Git commit SHA.
- Python version and environment.
- Input literature JSONL.
- Optional dataset catalog JSONL.
- Output directory.
- `litdatamatcher.sqlite`.
- `metrics.jsonl`.
- Expert review labels, if available.

## Extending Data Sources

Add a new dataset adapter by implementing the `DataSourceAdapter` protocol:

```python
class MyAdapter:
    name = "my_repository"

    def search(self, query: str) -> list[DatasetRecord]:
        ...
```

Adapters should preserve raw repository metadata in `DatasetRecord.metadata`
when possible. Live API adapters should include retry, caching, rate-limit,
schema-drift tests, source timestamps, and provenance before being used for
production ranking.

Optional scaffolds are already present for OpenAlex literature metadata and
ClinicalTrials.gov study metadata. Priority future adapters include PubMed, GEO,
SRA/ENA, MGnify, Qiita, Metabolomics Workbench, dbGaP, and domain-specific
biomedical repositories.

## Current Limitations

LitDataMatcher is ready as a research-software foundation, not as a validated
autonomous discovery engine. Important limitations remain:

- Open-question extraction is currently deterministic and should be benchmarked
  against expert annotations.
- Ranking weights are interpretable heuristics, not yet calibrated from expert
  relevance labels.
- The default dataset catalog is curated and offline; production use requires
  live source adapters and source-specific validation.
- The meta-analysis node currently estimates recurrence and evidence strength;
  it does not yet extract effect sizes or perform statistical meta-analysis.
- Top-ranked pairs require manual review for study design, consent, license,
  variable compatibility, confounding, missingness, and statistical power.

## Path To Publication Readiness

The next scientific milestones are:

1. Build an expert-annotated benchmark corpus for open questions and
   question-dataset relevance.
2. Report extraction precision, recall, F1, calibration, and error taxonomy.
3. Add live adapters for at least two major repositories.
4. Calibrate ranking with expert labels and report precision@k, MRR, and nDCG.
5. Produce case studies where ranked question-data opportunities are manually
   validated by domain experts.
6. Archive all inputs, outputs, environment metadata, and code at a stable
   release commit.

## Scientific Caveat

LitDataMatcher ranks opportunities. It does not prove that a public dataset can
answer a question. Its purpose is to accelerate expert review by making
question-data connections explicit, traceable, and reproducible.
