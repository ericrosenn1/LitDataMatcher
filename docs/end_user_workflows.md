# End-User Workflows

This document gives practical command-level workflows for growing
LitDataMatcher from local experiments toward reviewable research corpora.

Current maturity: LitDataMatcher provides local, deterministic,
provenance-aware research-software scaffolding. It is not yet a validated
biomedical discovery engine, not yet calibrated on a large expert-labeled
corpus, and not yet validated at scale on large PMC/database corpora.

For the source/provenance contract behind these workflows, see
`docs/source_ingestion_and_provenance.md`.

## Command Scope Summary

| Workflow | When to use it | Consumes | Produces | Does not do | Live API or service? | Output depth |
| --- | --- | --- | --- | --- | --- | --- |
| `ingest` | Convert local source files into pipeline-ready literature JSONL | local text, Markdown, PDF, JATS/PMC XML, GROBID TEI XML, generic XML, JSONL/NDJSON | literature JSONL, manifest JSON, ingestion report | does not extract questions, rank matches, run GROBID, or validate licenses | no live API; PDF uses local `pdfminer.six` if installed | full text, extracted PDF text, structured XML text, partial XML text, or record passthrough depending on input |
| `grobid-tei` | Convert one PDF to TEI XML before ingestion | local PDF and a configured GROBID endpoint | TEI XML file | does not ingest, extract questions, rank matches, or manage a GROBID server | calls a running/available GROBID service | TEI XML whose quality depends on the PDF and GROBID output |
| `literature-search` | Retrieve optional literature metadata candidates | query string and selected literature adapters | JSONL literature-like rows | does not guarantee full text or publication-ready evidence | can call live APIs unless cache supplies response | PubMed abstract plus metadata or metadata-only; OpenAlex metadata plus reconstructed abstract when available |
| `dataset-search` | Retrieve optional dataset metadata candidates | query string and selected dataset adapters | dataset JSONL | does not download/analyze datasets or verify analysis-ready variables | can call live APIs unless cache supplies response | dataset or clinical-trial metadata only |
| `capability-export` | Catalog observed and plausibly derived variables from dataset records | dataset JSONL | capability JSONL | does not compute derived variables or run statistical analyses | no live API | observed capability metadata and derived capability catalog claims |
| `review-queue` | Create a small recurring scoring sheet | completed run directory | CSV or JSONL queue | does not provide a web UI or adjudicate labels | no live API | review task rows with match/provenance context |
| `annotation-export` | Normalize completed review labels for QA/training | completed review CSV/JSONL | label JSONL, QA artifacts, optional splits, manifest | does not create gold labels by itself or train a model | no live API | human-label artifacts with preserved source metadata |
| `calibrate-ranking` | Summarize ranking thresholds from labels | matches JSONL and match-label JSONL | calibration report JSON | does not train a ranking model | no live API | threshold QA/reporting over existing labels |
| `stress-demo` | Run deterministic synthetic workflow checks | requested synthetic document count | synthetic corpus and run artifacts | does not validate real-world parsing or live adapters | no live API | synthetic demo data only |

## 1. Ingest Local Literature

Before expanding to large corpora, run a controlled smoke test with 3-5 files:

```bash
litdatamatcher manual-smoke --prepare-only
```

Place a few `.txt`, `.md`, `.jsonl`, `.pdf`, `.xml`, `.nxml`, or `.tei` files in
the prepared input folder, then run:

```bash
litdatamatcher manual-smoke
```

Inspect `manual_review_notes.md`, `smoke_test_summary.md`, the ingestion
manifest, extracted questions, ranked matches, and review sheet before adding
more parser complexity or scaling up.

Use `ingest` for local text, Markdown, PDF, JATS/PMC XML, GROBID TEI XML, or
existing JSONL records:

```bash
litdatamatcher ingest --input papers/ --out run/corpus/literature.jsonl --recursive --on-error skip
```

Review:

- `run/corpus/literature.jsonl`
- `run/corpus/literature.manifest.json`
- `run/corpus/literature.ingestion_report.md`

The JSONL records and ingestion report include source provenance metadata. Use
it to distinguish structured full text, extracted PDF text, record passthroughs,
abstract-only records, and dataset metadata before treating extracted questions
as equally well supported.

For PDFs, install the optional NLP dependencies. For higher-quality PDF
structure, run a local GROBID service, convert PDFs to TEI, and ingest the TEI
XML:

```bash
litdatamatcher grobid-tei --input papers/example.pdf --out run/tei/example.tei.xml
litdatamatcher ingest --input run/tei/example.tei.xml --out run/corpus/literature.jsonl
```

JATS/PMC XML and GROBID TEI ingestion preserve basic article metadata, section
headings, section records, and compact author names when those fields are
available in the XML.

## 2. Run The Canonical Pipeline

```bash
litdatamatcher run --input run/corpus/literature.jsonl --out run/full --top-n 100
```

Primary review artifacts:

- `run/full/matches.jsonl`
- `run/full/source_provenance_summary.json`
- `run/full/review_sheet.csv`
- `run/full/review_sheet.jsonl`
- `run/full/publication_report.md`

The review sheet includes compact provenance fields such as source type,
content scope, acquisition method, parser/adapter caveats, and source warnings.
The JSONL review export preserves the full nested provenance block for training
label export.

## 3. Search Optional Live Metadata Sources

Live adapter commands are optional and cached. They supplement, but do not
replace, the deterministic offline catalog.

```bash
litdatamatcher literature-search --query "microbiome antibiotic recovery" --source pubmed openalex --out run/search/literature.jsonl
litdatamatcher dataset-search --query "IBD transcriptomics treatment response" --source geo clinicaltrials mgnify --out run/search/datasets.jsonl
```

Treat live-source output as metadata candidates requiring validation before use
in a publication run.

Current live-source notes:

- PubMed uses NCBI ESearch/ESummary for discovery and citation metadata, then
  EFetch XML for abstracts and article identifiers when available.
- MGnify uses the API v2 studies endpoint and reads the v2 `items` list shape;
  legacy cached JSON:API-shaped rows remain supported for tests and older local
  artifacts.

## 4. Export Dataset Capabilities

```bash
litdatamatcher capability-export --datasets run/full/datasets.jsonl --out run/full/capabilities.jsonl
```

The capability file separates observed variables from plausibly derived
capabilities such as treatment response, survival-style outcomes, longitudinal
change, or BMI. These are catalog claims, not computed analyses.

## 5. Create Daily Review Queues

```bash
litdatamatcher review-queue --run-dir run/full --out run/full/daily_queue.csv --limit 5 --reviewer-id reviewer_a
```

The queue is intentionally simple: it is a small scoring sheet that can be
completed repeatedly by one or more reviewers. Completed queues can be passed to
`annotation-export`.

## 6. Build Annotation Corpora

```bash
litdatamatcher annotation-export --labels run/full/daily_queue.csv --out run/full/annotations --split-strategy by_source_id
```

Inspect:

- `annotation_corpus_report.md`
- `agreement_summary.json`
- `adjudication_needed.jsonl`
- `manifest.json`

Do not treat labels as gold-standard data until reviewer disagreements and
adjudication records have been resolved.

## 7. Calibrate Ranking Thresholds

After labels exist, create an exploratory calibration report:

```bash
litdatamatcher calibrate-ranking \
  --matches run/full/matches.jsonl \
  --labels run/full/annotations/question_data_match_labels.jsonl \
  --out run/full/annotations/ranking_calibration.json
```

This report chooses a simple threshold and summarizes score bins. It is not a
trained predictive model; use it as a first QA loop before richer training.

## 8. Run Stress Checks Before Large Corpora

```bash
litdatamatcher stress-demo --out run/stress_demo --documents 50 --top-n 100
```

The stress demo is synthetic and deterministic. It is useful for regression and
artifact checks before running large local PMC or GROBID-derived corpora.

## 9. Publication-Readiness Path

Recommended order:

1. Ingest a small real corpus with manifests and reports.
2. Run the canonical pipeline and inspect question wording.
3. Create daily review queues and export annotation artifacts.
4. Resolve adjudication-needed labels.
5. Evaluate extraction and ranking against gold labels.
6. Calibrate ranking or training loops only after label quality is stable.
7. Archive inputs, manifests, outputs, environment metadata, and reports.
