# Source Ingestion And Provenance

This document defines the source-ingestion and provenance contract for
LitDataMatcher. It is a practical guide for adding future parsers and adapters
without overstating what a source record can support.

## 1. Purpose

Source provenance exists to keep evidence depth visible throughout the workflow.
A record derived from structured full text, an abstract-only literature record,
a metadata-only dataset summary, and a derived capability catalog entry should
not be interpreted as equivalent.

The provenance layer supports:

- auditability across local files, API records, derived catalog entries, and
  synthetic/demo records
- reviewer interpretation of source strength, missingness, and parser caveats
- annotation and training-label exports that retain source context
- publication reports that describe what kinds of inputs supported a run
- reproducibility through stable source identifiers, local paths, URLs, content
  hashes, retrieval metadata, warnings, and limitations

The most important safety function is preventing weak source records from being
overinterpreted as strong evidence. For example, a PubMed abstract can be useful
for discovery, but it is not the same as a full-text article. A GEO record can
describe a dataset, but it is not the same as downloaded, inspected, and
analysis-ready data.

## 2. Source Types Currently Supported

| Source type | Expected input | Module | Expected output | Typical content scope | Key provenance fields | Known limitations | Recommended downstream use |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Local text | `.txt` or `.text` file | `litdatamatcher.ingestion` | literature JSONL record | full text if the file contains full text; otherwise partial text | `local_path`, `source_sha256`, `parser_name`, `content_scope`, `warnings` | section structure is inferred heuristically | small local corpora, draft smoke tests, converted text |
| Local Markdown | `.md` or `.markdown` file | `litdatamatcher.ingestion` | literature JSONL record | full text if supplied | same as local text | headings may not map cleanly to article sections | curated local notes or converted articles |
| Local JSONL/NDJSON | existing `.jsonl` or `.ndjson` records | `litdatamatcher.ingestion` | literature JSONL record(s) | record passthrough; depth depends on upstream fields | `local_path`, `source_sha256`, `record_count`, upstream IDs | upstream provenance may be incomplete | importing prepared literature records |
| Local PDF | `.pdf` file | `litdatamatcher.ingestion` with `pdfminer.six` | literature JSONL record | extracted full text, not guaranteed structured full text | `parser_name=pdfminer`, `local_path`, `source_sha256`, parser limitations | text order, captions, tables, references, and columns can be noisy; no OCR | early smoke tests when no XML/TEI is available |
| JATS/PMC XML | `.xml`, `.nxml`, or `.jats` article XML | `litdatamatcher.literature_xml` | literature JSONL record | structured full text when article body is present | `parser_name=jats`, IDs, sections, authors, journal, `source_sha256` | schema validation is not yet performed; article variants may omit fields | preferred full-text path for PMC-style articles |
| GROBID TEI XML | TEI XML already produced by GROBID | `litdatamatcher.literature_xml` | literature JSONL record | structured full text if GROBID output is complete | `parser_name=grobid_tei`, TEI IDs, sections, authors | quality depends on PDF quality and GROBID extraction | full-text PDF workflow after explicit TEI generation |
| Generic XML | unsupported XML structure | `litdatamatcher.literature_xml` | literature JSONL record | partial or shallow XML text | `parser_name=generic_xml`, root tag, local path | may miss article-specific metadata and section structure | fallback ingestion for inspection, not high-confidence extraction |
| PubMed / NCBI E-utilities | search query through optional adapter | `litdatamatcher.adapters.PubMedLiteratureAdapter` | literature JSONL-style metadata rows | abstract plus metadata when EFetch has an abstract; otherwise metadata only | `source_url`, `raw_record_id`, `adapter_name`, `content_scope` | usually not full text; live behavior is not validated at scale | literature discovery and abstract-level triage |
| OpenAlex | search query through optional adapter | `litdatamatcher.adapters.OpenAlexLiteratureAdapter` | literature JSONL-style metadata rows | metadata plus reconstructed abstract when available | `source_url`, `adapter_name`, concepts, DOI | reconstructed abstracts are not article text; coverage varies | broad scholarly discovery and metadata enrichment |
| Offline curated catalog | bundled deterministic records | `litdatamatcher.datasets.CuratedBiomedicalCatalogAdapter` | `DatasetRecord` | dataset metadata | dataset ID, source URL, catalog adapter, validation caveats | not a live repository snapshot; variables are curated summaries | deterministic demos and transparent local matching |
| ClinicalTrials.gov | search query through optional adapter | `litdatamatcher.adapters.ClinicalTrialsDatasetAdapter` | `DatasetRecord` | clinical trial metadata | NCT ID, source URL, API adapter, limitations | not patient-level data; registry fields require study-specific review | identifying potentially relevant trial records |
| GEO | search query through optional adapter | `litdatamatcher.adapters.GEODatasetAdapter` | `DatasetRecord` | dataset metadata | accession, source URL, API adapter, limitations | summary metadata may not expose all variables or files | identifying candidate omics datasets for later inspection |
| MGnify | search query through optional adapter | `litdatamatcher.adapters.MGnifyDatasetAdapter` | `DatasetRecord` | microbiome dataset metadata | accession, source URL, API adapter, sample count if available | list metadata may need study-detail follow-up | identifying candidate microbiome studies |
| Capability registry | `DatasetRecord` JSONL | `litdatamatcher.capability_registry` | `DatasetCapability` records | observed and plausibly derived capability catalog | dataset ID, source variables, rule ID, confidence, limitations | derived capabilities are not computed analyses | cataloging what may be possible before analysis |
| Synthetic/demo records | built-in demo or stress helper | `litdatamatcher.pipeline`, `litdatamatcher.stress` | literature records and run artifacts | synthetic demo text | synthetic source IDs; provenance may be absent or shallow | not real scientific evidence | regression, examples, and workflow checks only |

### Practical Source Profiles

These profiles restate the table as operational guidance for future adapters,
parsers, review exports, and reports.

#### Local Text

- Source type: local text.
- Input: `.txt` or `.text` files.
- Module: `litdatamatcher.ingestion`.
- Output: literature JSONL record.
- Content scope: full text only if the file itself contains full article text;
  otherwise partial local text.
- Strengths: simple, reproducible, hashable, and useful for small curated
  corpora.
- Limitations: section boundaries and article metadata are inferred from plain
  text.
- Provenance warnings: `missing_title`, `missing_abstract`,
  `missing_body_text`, or `low_traceability` when source text is shallow.
- Appropriate downstream use: small local smoke tests and manually curated
  corpora.
- What not to infer: do not infer that a text file is a peer-reviewed full-text
  article unless the source and content make that clear.

#### Local Markdown

- Source type: local Markdown.
- Input: `.md` or `.markdown` files.
- Module: `litdatamatcher.ingestion`.
- Output: literature JSONL record.
- Content scope: full text or partial text depending on the Markdown content.
- Strengths: readable local format with headings that may help sectioning.
- Limitations: headings may be notes or report structure rather than article
  sections.
- Provenance warnings: `parser_fallback`, `missing_abstract`, or
  `low_traceability` when Markdown is not a direct article source.
- Appropriate downstream use: curated local notes, converted manuscripts, and
  small review corpora.
- What not to infer: do not infer article structure or publication status from
  Markdown formatting alone.

#### Local JSONL / NDJSON

- Source type: local JSONL.
- Input: `.jsonl` or `.ndjson` records with literature-like fields.
- Module: `litdatamatcher.ingestion`.
- Output: one or more literature JSONL records.
- Content scope: record passthrough; depth depends on upstream fields.
- Strengths: preserves prepared records and can carry upstream IDs.
- Limitations: upstream provenance may be incomplete or inconsistent.
- Provenance warnings: `low_traceability`, `missing_body_text`, or
  `missing_abstract` if required context is absent.
- Appropriate downstream use: importing prepared corpora or search outputs.
- What not to infer: do not assume JSONL records are full text unless `text`
  and provenance indicate that.

#### Local PDF

- Source type: local PDF.
- Input: `.pdf` files.
- Module: `litdatamatcher.ingestion` with optional `pdfminer.six`.
- Output: literature JSONL record.
- Content scope: extracted PDF text, not guaranteed structured full text.
- Strengths: useful when no XML or TEI representation is available.
- Limitations: no OCR; columns, tables, captions, references, and reading order
  can be noisy.
- Provenance warnings: `parser_fallback`, `missing_body_text`,
  `missing_abstract`, or `low_traceability` when extraction is weak.
- Appropriate downstream use: early inspection and smoke tests.
- What not to infer: do not treat extracted PDF text as clean structured full
  text without manual review.

#### JATS / PMC XML

- Source type: JATS/PMC XML.
- Input: `.xml`, `.nxml`, or `.jats` article XML.
- Module: `litdatamatcher.literature_xml`.
- Output: literature JSONL record.
- Content scope: structured full text when article body is present.
- Strengths: often preserves article metadata, abstract, body sections,
  article IDs, journal, and authors.
- Limitations: schema validation is not yet performed; article variants may
  omit expected fields.
- Provenance warnings: `missing_body_text`, `missing_abstract`, or
  `parser_fallback` if XML structure is shallow.
- Appropriate downstream use: preferred local full-text path for PMC-style
  articles.
- What not to infer: do not infer schema validity or license compliance from
  successful parsing alone.

#### GROBID TEI XML

- Source type: GROBID TEI XML.
- Input: TEI XML already produced by GROBID.
- Module: `litdatamatcher.literature_xml`.
- Output: literature JSONL record.
- Content scope: structured full text if the GROBID output contains usable body
  text.
- Strengths: can recover structured article metadata and sections from PDFs.
- Limitations: quality depends on PDF quality and upstream GROBID extraction.
- Provenance warnings: `parser_fallback`, `missing_body_text`,
  `missing_abstract`, or `low_traceability` when TEI is incomplete.
- Appropriate downstream use: PDF-derived full-text workflow after explicit TEI
  generation.
- What not to infer: ingesting TEI does not mean LitDataMatcher automatically
  ran or managed a GROBID server.

#### Generic XML

- Source type: generic XML.
- Input: unsupported XML structure.
- Module: `litdatamatcher.literature_xml`.
- Output: literature JSONL record.
- Content scope: partial or shallow XML text.
- Strengths: provides a conservative fallback for inspection.
- Limitations: may miss article-specific metadata, sections, and IDs.
- Provenance warnings: `parser_fallback`, `missing_title`,
  `missing_abstract`, `missing_body_text`, or `low_traceability`.
- Appropriate downstream use: inspection and triage before writing a dedicated
  parser.
- What not to infer: do not treat generic XML parsing as equivalent to JATS or
  GROBID TEI parsing.

#### PubMed / NCBI E-utilities

- Source type: PubMed.
- Input: query through the optional PubMed adapter.
- Module: `litdatamatcher.adapters.PubMedLiteratureAdapter`.
- Output: literature JSONL-style metadata rows.
- Content scope: abstract plus metadata when EFetch provides an abstract;
  otherwise metadata only.
- Strengths: useful for discovery, citation metadata, PMID/DOI tracking,
  journal metadata, and abstract-level triage.
- Limitations: not equivalent to full-text evidence extraction; live behavior
  has not been validated at scale.
- Provenance warnings: `abstract_only`, `metadata_only`, `missing_abstract`,
  or `no_live_validation` as applicable.
- Appropriate downstream use: literature discovery and abstract-level
  candidate generation.
- What not to infer: do not infer full text, full methods, or complete
  limitation/future-work context unless full text is separately available.

#### OpenAlex

- Source type: OpenAlex.
- Input: query through the optional OpenAlex adapter.
- Module: `litdatamatcher.adapters.OpenAlexLiteratureAdapter`.
- Output: literature JSONL-style metadata rows.
- Content scope: scholarly metadata plus reconstructed abstract when available.
- Strengths: useful for discovery, linking, citation context, DOI metadata, and
  concept metadata.
- Limitations: reconstructed abstracts are not article body text; coverage
  varies.
- Provenance warnings: `metadata_only`, `abstract_only`, or
  `no_live_validation` when appropriate.
- Appropriate downstream use: metadata enrichment and broad scholarly search.
- What not to infer: do not treat OpenAlex records as article full text.

#### ClinicalTrials.gov

- Source type: ClinicalTrials.gov.
- Input: query through the optional ClinicalTrials.gov adapter.
- Module: `litdatamatcher.adapters.ClinicalTrialsDatasetAdapter`.
- Output: `DatasetRecord`.
- Content scope: clinical trial registry metadata.
- Strengths: useful for study design, intervention, condition, status, and
  outcome metadata.
- Limitations: not patient-level trial data; trial metadata does not guarantee
  data availability or reuse rights.
- Provenance warnings: `metadata_only`, `license_unknown`, or
  `no_live_validation` when appropriate.
- Appropriate downstream use: identifying candidate trial records and study
  design context.
- What not to infer: do not infer individual-level access, consent clearance,
  or analyzable trial data from registry metadata alone.

#### GEO

- Source type: GEO.
- Input: query through the optional GEO adapter.
- Module: `litdatamatcher.adapters.GEODatasetAdapter`.
- Output: `DatasetRecord`.
- Content scope: dataset or study metadata.
- Strengths: useful for accession-level discovery and high-level assay/sample
  context.
- Limitations: not automatically analysis-ready expression data; requires
  accession-level inspection, download, normalization, and downstream
  processing.
- Provenance warnings: `metadata_only`, `missing_abstract`,
  `license_unknown`, or `no_live_validation` as applicable.
- Appropriate downstream use: candidate dataset discovery before data access
  and manual review.
- What not to infer: do not infer that variables, covariates, timing, or
  missingness are adequate from summary metadata alone.

#### MGnify

- Source type: MGnify.
- Input: query through the optional MGnify adapter.
- Module: `litdatamatcher.adapters.MGnifyDatasetAdapter`.
- Output: `DatasetRecord`.
- Content scope: microbiome study or dataset metadata.
- Strengths: useful for microbiome study discovery, accession tracking, and
  coarse sample/assay context.
- Limitations: not automatically an analysis-ready sample table; source-specific
  download and metadata validation are still required.
- Provenance warnings: `metadata_only`, `missing_body_text`,
  `license_unknown`, or `no_live_validation` when appropriate.
- Appropriate downstream use: candidate microbiome dataset discovery.
- What not to infer: do not infer sample-level metadata completeness,
  taxonomic/functional table availability, or analysis-ready variables from
  list metadata alone.

#### Capability Registry Outputs

- Source type: capability registry.
- Input: normalized `DatasetRecord` objects.
- Module: `litdatamatcher.capability_registry`.
- Output: `DatasetCapability` records.
- Content scope: observed capability metadata and plausibly derived capability
  catalog entries.
- Strengths: makes possible observed and derived variables explicit for review.
- Limitations: not a computed analysis and not evidence of adequate timing,
  quality, units, or missingness.
- Provenance warnings: `derived_not_computed`, `metadata_only`, or
  `low_traceability` where supporting metadata is weak.
- Appropriate downstream use: planning, triage, and reviewer inspection of what
  might be computable.
- What not to infer: do not infer that a derived variable has already been
  calculated or is valid for analysis.

#### Offline Curated Dataset Catalog

- Source type: curated biomedical catalog.
- Input: built-in deterministic `DatasetRecord` summaries.
- Module: `litdatamatcher.datasets.CuratedBiomedicalCatalogAdapter`.
- Output: `DatasetRecord` objects with advisory `metadata.source_provenance`.
- Content scope: dataset metadata.
- Strengths: reproducible offline matching and transparent demo coverage.
- Limitations: not a live database snapshot; repository-specific validation is
  still required.
- Provenance warnings: catalog metadata should be checked against source
  repositories before publication use.
- Appropriate downstream use: deterministic local demos, review of matching
  behavior, and early feature development.
- What not to infer: do not infer source completeness, downloaded data, or
  analysis-ready variables.

#### Synthetic / Demo Records

- Source type: synthetic.
- Input: built-in demo records or stress-helper generated records.
- Module: `litdatamatcher.pipeline` or `litdatamatcher.stress`.
- Output: literature records and run artifacts.
- Content scope: synthetic demo text.
- Strengths: deterministic, lightweight, useful for workflow and artifact
  checks.
- Limitations: not real literature, not real datasets, and not scientific
  evidence.
- Provenance warnings: `synthetic`, `low_traceability`, or `not_validated_live`
  depending on context.
- Appropriate downstream use: examples, demonstrations, and local regression
  checks.
- What not to infer: do not infer real biological validity or publication
  readiness from synthetic examples.

## 3. Recommended Provenance Vocabulary

The current code accepts free-text provenance values. Future adapters and
parsers should converge toward the controlled values below. Some current values
are more implementation-specific, such as `text`, `markdown`, `pdf`, `jats`,
`grobid_tei`, `generic_xml`, `full_text_extracted`,
`metadata_plus_reconstructed_abstract`, and `study_metadata`; these should be
mapped or documented rather than renamed casually.

### `source_type`

Recommended values:

- `local_text`
- `local_markdown`
- `local_jsonl`
- `local_pdf`
- `jats_xml`
- `grobid_tei`
- `generic_xml`
- `pubmed`
- `openalex`
- `clinicaltrials`
- `geo`
- `mgnify`
- `curated_biomedical_catalog`
- `capability_registry`
- `synthetic`

### `content_scope`

Recommended values:

- `full_text`
- `abstract_only`
- `metadata_only`
- `dataset_metadata`
- `clinical_trial_metadata`
- `derived_capability_catalog`
- `synthetic_demo`

When needed, adapters may add a more specific value, but documentation should
explain how it maps to these categories.

### `acquisition_method`

Recommended values:

- `local_file`
- `local_directory`
- `api`
- `cached_api`
- `grobid_upload`
- `manual`
- `synthetic`

### `parser_name`

Recommended values:

- `local_text_parser`
- `pdfminer`
- `jats_xml_parser`
- `grobid_tei_parser`
- `generic_xml_parser`

### `adapter_name`

Recommended values:

- `pubmed_eutilities`
- `openalex`
- `clinicaltrials`
- `geo`
- `mgnify`

### `status`

Recommended values:

- `parsed`
- `skipped`
- `partial`
- `failed`
- `mock_or_cached`
- `not_validated_live`

Current helper defaults may use `ok`; documentation and reports should treat
that as a successful parse or adapter normalization, not as scientific
validation.

### Warning Categories

Recommended warning categories:

- `abstract_only`
- `metadata_only`
- `missing_title`
- `missing_abstract`
- `missing_body_text`
- `parser_fallback`
- `no_live_validation`
- `derived_not_computed`
- `license_unknown`
- `low_traceability`

Warnings should name what is weak or missing. They should not imply that the
record is unusable.

## 4. Provenance Lifecycle

Expected flow:

```text
source file or API response
  -> SourceProvenance
  -> literature record or dataset record
  -> QuestionCandidate metadata
  -> MatchCandidate or review export
  -> annotation export
  -> source_provenance_summary.json
  -> publication report
```

At acquisition time, local file metadata or remote adapter metadata is turned
into `SourceProvenance`. This should include source identity, locator, content
scope, acquisition method, parser/adapter identity, native IDs, caveats, and
timestamps when available.

At literature or dataset record creation, provenance is stored as
`source_provenance` on the record or under `metadata.source_provenance`. Raw
source metadata should remain available where feasible.

At question extraction, literature provenance is copied into
`QuestionCandidate.metadata.source_provenance`. The extracted evidence spans
preserve source IDs, DOI, title, section, and sentence index when available.
Character offsets are not yet consistently preserved.

At matching and review export, CSV exports expose compact provenance columns
for question/literature provenance and dataset/catalog provenance separately.
Use `question_source_*` fields for the literature side and
`dataset_source_*` fields for the matched data side; the older `source_*`
columns are retained as question-source compatibility fields. JSONL review
exports preserve both side-specific top-level provenance fields and the nested
match object.

At annotation export, completed review labels preserve available source IDs,
document IDs, evidence metadata, and source provenance in label metadata. This
allows future training and calibration to account for source quality.

At reporting, `source_provenance_summary.json` and publication reports summarize
source-type, content-scope, acquisition-method, status, warning, and limitation
counts. These summaries are audit aids, not claims of source validity.

`module_boundary_map.json` and `provenance_transfer_check.json` are advisory
developer diagnostics. They make module ownership and provenance visibility
inspectable, but they do not validate scientific correctness or block a run.

What may be lost:

- exact character offsets through PDF/XML extraction
- all native API fields if an adapter intentionally normalizes to a small subset
- raw XML structure beyond stored section records
- distinction between cached and live responses unless the adapter records it
- complete licensing or consent context when source metadata does not expose it

## 5. Confidence Implications

Source type should affect interpretation:

- JATS/PMC XML full text is usually stronger than abstract-only PubMed
  metadata because it can expose full discussion, limitation, and methods text.
- GROBID TEI can be strong, but quality depends on PDF quality and GROBID
  output. A clean TEI file is not guaranteed just because a PDF was submitted.
- Generic XML may be structurally shallow and should be treated as fallback
  ingestion unless manually inspected.
- PubMed is useful for literature discovery and abstract metadata, but it is
  not always enough for full-text question extraction.
- OpenAlex is useful for scholarly metadata and discovery, but it is not a
  substitute for article text.
- ClinicalTrials.gov records provide protocol and trial metadata, not
  patient-level trial data.
- GEO records describe datasets, but matching still needs manual metadata
  inspection and eventual data access.
- MGnify records describe microbiome studies and datasets, but do not by
  themselves establish analysis-ready variables.
- Derived capabilities indicate potential computability, not that an analysis
  has been run.

Reviewers should be especially cautious when a high-ranked match is supported
only by abstract-only or metadata-only provenance.

## 6. Reporting Semantics

`source_provenance_summary.json` currently summarizes provenance entries found
on question and dataset records written by a pipeline run. Counts are by
provenance entry, not necessarily by unique paper, unique dataset, or unique
source repository.

The summary currently counts:

- `records_with_provenance`: number of provenance entries observed
- `source_types`: count by source type value
- `content_scopes`: count by content scope value
- `acquisition_methods`: count by acquisition method value
- `statuses`: count by status value
- `warnings`: top warning strings
- `limitations`: top limitation strings

The publication report provenance section should imply only that a run contains
records with those provenance categories. It should not imply that live APIs
were validated, records are complete, licenses are cleared, datasets are
downloaded, variables are analysis-ready, or derived capabilities were computed.

Review-facing outputs should describe abstract-only and metadata-only records
plainly. For example:

- "PubMed abstract plus metadata" should not be shortened to "full-text paper."
- "GEO dataset metadata" should not be shortened to "analyzed GEO dataset."
- "Derived capability catalog" should not be shortened to "computed outcome."

Known ambiguity: current provenance values are still partly implementation
specific. Reports should preserve the recorded value and docs should explain
the interpretation until a stricter vocabulary migration is chosen.

## 7. Adapter Contract

Every future adapter should:

- attach `SourceProvenance`
- declare `source_type`
- declare `content_scope`
- declare `acquisition_method`
- preserve native source identifiers
- preserve retrieval URL or query when available
- preserve cache metadata when applicable
- preserve retrieval timestamp when applicable
- record parser or adapter name and version if known
- record warnings and limitations
- avoid claiming full text when only metadata or abstracts are present
- avoid claiming analysis-ready datasets from metadata-only records
- avoid overstating derived capabilities
- use stable `document_id` and `source_id` conventions where applicable
- produce deterministic output for cached or mocked tests

Adapter outputs should remain normalized enough for the pipeline, while keeping
raw repository metadata under `metadata` whenever practical.

## 8. Parser Contract

Every parser should:

- produce stable `source_id` and `document_id` where possible
- preserve title, abstract, body text, authors, year, journal, DOI, PMCID,
  accession, or equivalent native IDs when available
- preserve section metadata when available
- explicitly declare whether it extracted full text, partial text, metadata
  only, or failed
- report parser fallback behavior
- never silently convert failed parsing into high-confidence body text
- record warnings for missing body text or weak structure

Parsers should prefer conservative output over confident but misleading output.
If structure is uncertain, the provenance warning should say so.

## 9. Examples

### Example A: JATS/PMC XML Full-Text Article

A PMC-style `.nxml` file contains article metadata, an abstract, body sections,
DOI, PMCID, authors, and journal title. Expected provenance is a structured
full-text record with a JATS parser name, local path, file hash, and caveat that
schema validation has not yet been performed. Downstream confidence is stronger
than abstract-only metadata, but evidence spans still need review.

Illustrative provenance:

```json
{
  "source_type": "jats_xml",
  "content_scope": "full_text",
  "acquisition_method": "local_file",
  "parser_name": "jats_xml_parser",
  "warnings": []
}
```

### Example B: GROBID TEI XML From A PDF

A PDF has already been processed by GROBID, producing TEI XML. LitDataMatcher
ingests the TEI file; it does not infer that GROBID was automatically run during
ingestion. Expected provenance should say GROBID TEI, local file, structured
full text if body text exists, and a limitation that quality depends on the
upstream PDF and GROBID extraction.

### Example C: PubMed Abstract-Only Record

A PubMed adapter record may include PMID, title, DOI, journal, authors, and an
abstract from EFetch XML. Expected provenance is `pubmed` with abstract plus
metadata or metadata only. Downstream extraction may identify useful questions,
but reviewers should not treat those questions as full-text literature mining.

### Example D: GEO Dataset Metadata Record

A GEO adapter record may include accession, title, summary, sample count, assay
type, and repository URL. Expected provenance is dataset metadata, not
downloaded data. Downstream matching can suggest relevance, but manual file and
metadata inspection is still required before analysis.

### Example E: Derived Capability Registry Entry

If a dataset contains height and weight fields, the capability registry may
emit a plausible `body_mass_index` capability. Provenance should describe this
as a derived capability catalog claim. It means BMI may be computable if units
and missingness are suitable; it does not mean BMI was computed.

## 10. Known Limitations And Future Work

- no large-corpus validation yet
- live adapters need real API validation
- no direct PMC bulk workflow yet
- no automatic GROBID server orchestration beyond the optional bridge command
- limited section-offset and evidence-span traceability
- no OCR
- no license-compliance layer beyond stored metadata and warnings unless added
  elsewhere
- review workflow remains local CSV/JSONL rather than a web UI
- calibration is a threshold QA/reporting step, not trained ranking
- current provenance vocabulary is not yet strictly enforced

## Checklist For Adding A New Source Adapter

- Identify the source repository or service.
- Declare the source identity and stable native IDs.
- Declare content scope: full text, abstract-only, metadata-only, dataset
  metadata, derived catalog, or synthetic.
- Declare acquisition method: API, cached API, local file, manual, or synthetic.
- Preserve retrieval URL, query, accession, DOI, PMID, PMCID, NCT ID, or other
  native IDs when available.
- Attach `SourceProvenance`.
- Preserve useful raw source metadata under `metadata`.
- Record warnings and limitations.
- Add mocked or cached deterministic tests when code changes are made.
- Avoid overstating full text, answerability, data access, or analysis-ready
  status.
- Check review/export behavior so provenance survives to labels.
- Update documentation for the new source and its caveats.

## Checklist For Adding A New Parser

- Identify the file type and expected extensions.
- Document the parser dependency and whether it is optional.
- Define fallback behavior for partial or failed parsing.
- Preserve extracted fields: title, abstract, body text, authors, year, journal,
  DOI, PMCID, accession, and equivalent native IDs when available.
- Preserve section metadata when available.
- Keep `source_id` and `document_id` stable.
- Report errors without silently creating high-confidence body text.
- Add provenance warnings for missing body text, missing abstract, weak
  structure, parser fallback, or low traceability.
- Add tests and representative fixtures when code changes are made.
- Document license and provenance caveats.
