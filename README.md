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

The current implementation is a local research-software scaffold and
deterministic pipeline foundation. It is offline and reproducible by default,
produces auditable JSONL/SQLite artifacts, and is organized into explicit
scientific nodes that can be independently tested, benchmarked, replaced, or
extended. It is not yet a validated biomedical discovery engine, not yet
calibrated on a large expert-labeled corpus, and not yet validated at scale on
large PMC/database corpora.

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
- Optional live metadata adapters for PubMed/OpenAlex literature search and
  ClinicalTrials.gov/GEO/MGnify dataset discovery.
- Source/provenance-aware ingestion and review metadata that distinguish full
  text, abstract-only records, metadata-only records, dataset metadata, and
  derived capability catalog entries.
- Dataset capability and derived-variable export for observed versus plausibly
  derivable fields.
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
- Preserved legacy streaming and literature prototypes for provenance and
  future cannibalization.

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
- `workflows/legacy_streaming/`: archived streaming subprocess entrypoints used
  for earlier topic-level demonstrations and resource-management experiments.

The package path should be used for serious analyses. Legacy literature code is
preserved under `archive/legacy_literature/`, training experiments under
`training/`, and historical data/model artifacts under `data/legacy_training/`
and `models/legacy/`.

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
litdatamatcher run --input tests/fixtures/test_input.jsonl --out run/full --top-n 100
```

Each input row should include some combination of:

```json
{"title": "...", "abstract": "...", "text": "...", "doi": "..."}
```

Convert local text, Markdown, PDF, JATS/PMC XML, GROBID TEI XML, or existing
JSONL sources into that format:

```bash
litdatamatcher ingest --input papers/ --out run/corpus/literature.jsonl --recursive
litdatamatcher run --input run/corpus/literature.jsonl --out run/full --top-n 100
```

Ingestion writes three reviewable artifacts:

- `literature.jsonl`: canonical records for `litdatamatcher run`.
- `literature.manifest.json`: machine-readable file provenance, record counts,
  skipped-file diagnostics, source IDs, and document IDs.
- `literature.ingestion_report.md`: human-readable ingestion summary.

`source_path` records local file provenance. `source_id` and `document_id` are
stable content-based grouping keys that help review exports and training labels
trace records back to their ingested source.

Each ingested or remotely retrieved record can also include a
`source_provenance` object. This records source type, content depth, acquisition
method, parser/adapter name, retrieval time, source URL or local path, caveats,
and the next intended handoff. This distinction is important because a
structured full-text JATS/TEI record, an abstract-only PubMed record, and a
dataset-metadata MGnify record should not be interpreted as equivalent evidence
depth.

The source-ingestion and provenance contract is documented in
`docs/source_ingestion_and_provenance.md`.

PDF ingestion uses the optional `pdfminer.six` dependency included in the
`.[nlp]` installation extra. XML ingestion supports deterministic JATS/PMC and
GROBID TEI parsing from local files, including section records and compact
author metadata when present.

If you have a local GROBID service running, convert a PDF to TEI before
ingestion:

```bash
litdatamatcher grobid-tei --input papers/example.pdf --out run/tei/example.tei.xml
litdatamatcher ingest --input run/tei/example.tei.xml --out run/corpus/literature.jsonl
```

Search optional live metadata sources:

```bash
litdatamatcher literature-search --query "microbiome antibiotic recovery" --source pubmed --out run/search/pubmed.jsonl
litdatamatcher dataset-search --query "IBD transcriptomics treatment response" --source geo mgnify --out run/search/datasets.jsonl
```

The PubMed adapter uses NCBI E-utilities ESearch/ESummary plus EFetch XML so
abstracts and article identifiers can be preserved when available. The MGnify
adapter targets API v2 and still accepts legacy cached/mock JSON:API-shaped
rows for reproducible tests.

Export observed and derived dataset capabilities from a dataset JSONL:

```bash
litdatamatcher capability-export --datasets run/full/datasets.jsonl --out run/full/capabilities.jsonl
```

## Outputs

A full run writes:

- `questions.jsonl`: extracted and normalized open research questions.
- `datasets.jsonl`: normalized candidate dataset records.
- `syntheses.jsonl`: question-cluster evidence summaries.
- `matches.jsonl`: ranked question-dataset opportunities with component scores.
- `summary.md`: concise human-readable ranking summary.
- `publication_report.md`: manuscript-style methods/results summary.
- `source_provenance_summary.json`: machine-readable source-type,
  content-scope, acquisition, status, warning, and limitation counts.
- `module_boundary_map.json`: developer-facing ownership map for pipeline
  responsibilities.
- `provenance_transfer_check.json`: advisory traceability check showing whether
  provenance stayed visible across handoffs.
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

Export completed review labels as normalized annotation-training artifacts:

```bash
litdatamatcher annotation-export --labels run/full/review.csv --out run/full/annotations
```

The annotation export schema, optional split files, QA artifacts, and
training-readiness statuses are documented in
`docs/annotation_export_schema.md`.

Create an exploratory ranking-threshold calibration report from completed match
labels:

```bash
litdatamatcher calibrate-ranking ^
  --matches run/full/matches.jsonl ^
  --labels run/full/annotations/question_data_match_labels.jsonl ^
  --out run/full/annotations/ranking_calibration.json
```

Generate grouped train/validation/test split files for later calibration or
training experiments:

```bash
litdatamatcher annotation-export ^
  --labels run/full/review.csv ^
  --out run/full/annotations ^
  --split-strategy by_question_id ^
  --split-fractions 0.8 0.1 0.1
```

Generate a manuscript-style report:

```bash
litdatamatcher report --run-dir run/full --out run/full/publication_report.md
```

Create a small daily review queue:

```bash
litdatamatcher review-queue --run-dir run/full --out run/full/daily_queue.csv --limit 5 --reviewer-id reviewer_a
```

Run a deterministic stress smoke test:

```bash
litdatamatcher stress-demo --out run/stress_demo --documents 50 --top-n 100
```

Run a controlled 3-5 file manual smoke test before scaling real corpora:

```bash
litdatamatcher manual-smoke --prepare-only
litdatamatcher manual-smoke
```

## Streaming Orchestrator Demo

The legacy subprocess demo is preserved for provenance and future workflow
experiments:

```bash
python workflows/legacy_streaming/orchestrator.py --out run/orchestrator_matches.jsonl
```

Provide custom topics with:

```bash
python workflows/legacy_streaming/orchestrator.py --topics-file topics.txt --out run/orchestrator_matches.jsonl
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
- `litdatamatcher.provenance`: source provenance helpers and summary counts.
- `litdatamatcher.adapters`: optional live-source adapter scaffolds.
- `litdatamatcher.capability_registry`: observed and derived dataset capability
  inference.
- `litdatamatcher.literature_xml`: JATS/PMC, GROBID TEI, and generic XML
  ingestion.
- `litdatamatcher.review_queue`: small recurring review-queue exports.
- `litdatamatcher.stress`: deterministic synthetic corpus stress helpers.
- `litdatamatcher.manual_smoke`: controlled real-file smoke-test workflow.
- `litdatamatcher.pipeline`: end-to-end orchestration.
- `litdatamatcher.cli`: command-line interface.

Additional documentation:

- `docs/architecture.md`
- `docs/node_contracts.md`
- `docs/reproducibility.md`
- `docs/source_ingestion_and_provenance.md`
- `docs/model_robustness_plan.md`
- `docs/feature_roadmap.md`
- `docs/end_user_workflows.md`

## Testing

After installing development dependencies:

```bash
python -m pytest
python -m compileall litdatamatcher
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
- Optional HTTP adapters use caching, retry scaffolds, and deterministic tests
  with mocked network responses.
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

Optional adapters are already present for PubMed/OpenAlex literature metadata
and ClinicalTrials.gov/GEO/MGnify dataset metadata. Priority future adapters
include SRA/ENA, Qiita, Metabolomics Workbench, dbGaP, and domain-specific
biomedical repositories.

## Citation And License

Citation metadata is provided in `CITATION.cff`. If this project contributes to
academic work, cite the software repository and the exact commit or release used
for the analysis.

The project is distributed under the MIT License. See `LICENSE`.

## Current Limitations

LitDataMatcher is ready as a research-software foundation, not as a validated
autonomous discovery engine. Important limitations remain:

- Open-question extraction is currently deterministic and should be benchmarked
  against expert annotations.
- Ranking weights are interpretable heuristics, not yet calibrated from expert
  relevance labels.
- Source/provenance reporting distinguishes evidence depth, but it does not
  validate that all source records are complete or publication-ready.
- The default dataset catalog is curated and offline; production use requires
  live source adapters and source-specific validation. Its provenance records are
  advisory catalog metadata, not evidence that source datasets were downloaded.
- Optional live adapters have mocked/cached test coverage, but have not yet
  been validated at scale against real API behavior.
- Dataset metadata records are not downloaded or analyzed datasets, and derived
  capabilities are plausibility/catalog claims rather than computed results.
- Ranking calibration is currently a threshold QA/reporting utility, not a
  trained ranking model.
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
