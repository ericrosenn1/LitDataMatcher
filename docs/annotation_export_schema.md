# Annotation Export Schema

This document describes the local annotation-corpus artifacts produced by:

```bash
litdatamatcher annotation-export --labels completed_review.jsonl --out run/annotations
```

The exporter turns completed review CSV/JSONL files into normalized training
label files plus reproducibility and QA artifacts. It does not train a model,
adjudicate reviewer disagreements, or change the scientific pipeline.

## Export Inputs

`annotation-export` accepts one or more completed review files from
`review_sheet.csv`, `review_sheet.jsonl`, or compatible annotation tools.

Core row identifiers:

- `match_id`: ranked question-dataset match identifier.
- `question_id`: extracted question identifier.
- `dataset_id`: matched dataset identifier.
- `annotator_id`, `reviewer_id`, or `expert_id`: optional row-level reviewer ID.

Current label fields:

- `match_relevance`: reviewer judgment of question-data relevance.
- `expert_match_relevance` or `expert_relevance`: accepted legacy aliases.
- `expert_question_quality`: optional 0..5 question-quality score.
- `expert_data_match_quality`: optional 0..5 data-match quality score.
- `expert_notes`, `review_notes`, or `notes`: optional reviewer notes.

Optional grouping provenance:

- `source_id`, `primary_source_id`, or `source_ids`: source paper/text
  identifiers.
- `document_id` or `document_ids`: document identifiers.
- `evidence_dois`, `evidence_titles`, `evidence_sections`, and
  `evidence_sentence_indices`: compact evidence provenance used for inspection
  and future error analysis.
- Nested `match.question.source_ids` or `match.question.document_ids` in JSONL
  review records.

Source/document IDs are needed for leakage-resistant document-level train/test
splits. Review-file paths are stored as provenance but are not treated as
scientific source IDs.

## Output Directory

Each export writes:

- `question_data_match_labels.jsonl`
- `question_quality_scores.jsonl`
- `annotation_corpus_summary.json`
- `manifest.json`
- `annotation_corpus_report.md`
- `agreement_summary.json`
- `adjudication_needed.jsonl`
- `warnings.jsonl`
- `skipped_rows.jsonl`
- `duplicates.jsonl`
- `conflicts.jsonl`

When split generation is requested, the exporter also writes:

- `splits/train.jsonl`
- `splits/validation.jsonl`
- `splits/test.jsonl`

Split generation is off by default.

## Question-Data Match Labels

`question_data_match_labels.jsonl` contains one `QuestionDataMatchLabel` object
for each usable row with match-level label content.

Important fields:

- `label_id`: deterministic label identifier.
- `match_id`, `question_id`, `dataset_id`: stable object references.
- `annotator_id`: row-level reviewer ID, or CLI fallback if no row ID exists.
- `label`: `relevant`, `not_relevant`, `uncertain`, or `unlabeled`.
- `relevance_score`: normalized 0..1 match relevance.
- `question_quality_score`: optional 0..5 score carried from the review row.
- `data_match_quality_score`: optional 0..5 score carried from the review row.
- `answerability_score`: optional 0..5 field reserved for richer labels.
- `notes`: reviewer notes.
- `metadata`: export provenance and non-core fields.

Current metadata includes:

- `rank`: review-sheet rank when available.
- `score`: original model score when available.
- `source_review_file`: input review file path.
- `raw_match_relevance`: raw reviewer relevance value before normalization.
- Optional `source_id`, `primary_source_id`, `source_ids`, `document_id`, or
  `document_ids`.
- Optional `evidence_dois`, `evidence_titles`, `evidence_sections`, and
  `evidence_sentence_indices`.

## Question Quality Scores

`question_quality_scores.jsonl` contains one `QuestionQualityScore` object for
each usable row with `expert_question_quality`.

Important fields:

- `label_id`: deterministic label identifier.
- `question_id`: stable question reference.
- `annotator_id`: row-level reviewer ID, or CLI fallback.
- `overall_score`: normalized reviewer quality score on the 0..5 scale.
- `clarity_score`, `importance_score`, `novelty_score`, `actionability_score`,
  and `translational_score`: reserved optional fields for richer review forms.
- `notes`: reviewer notes.
- `metadata`: `rank`, `match_id`, `dataset_id`, `source_review_file`, and
  optional source/document/evidence provenance.

## Optional Split Files

Split files are generated only when `--split-strategy` is not `none`.

Example:

```bash
litdatamatcher annotation-export \
  --labels run/full/review.csv \
  --out run/full/annotations \
  --split-strategy by_question_id \
  --split-fractions 0.8 0.1 0.1
```

Supported strategies:

- `by_question_id`: keep all labels for the same question in one split.
- `by_source_id`: keep all labels for the same source in one split when source
  IDs are present.
- `by_document_id`: keep all labels for the same document in one split when
  document IDs are present.
- `random`: row-level deterministic split, available only when explicitly
  requested.

Grouping precedence:

- `by_source_id`: `source_id`, then `primary_source_id`, then `source_ids`,
  then `document_id`, then `document_ids`.
- `by_document_id`: `document_id`, then `document_ids`, then `source_id`, then
  `primary_source_id`, then `source_ids`.

Split files contain a combined stream of exported label rows. Each row has:

- `label_type`: `question_data_match` or `question_quality`.
- The original label fields for that label family.
- `metadata.split_group`: the actual group assigned to one split.
- `metadata.split_strategy`: the requested strategy.
- `metadata.split_grouping_field`: the field that actually controlled grouping.

If `by_source_id` or `by_document_id` is requested but source/document IDs are
missing, the splitter falls back to `question_id` when possible and records a
warning in `manifest.json`. If no usable ID exists, it falls back to a row index
and records a warning.

Review-file paths such as `metadata.source_review_file` are not scientific
source/document IDs and are not used as split grouping keys.

## Manifest

`manifest.json` is the main reproducibility contract for an export.

Top-level fields:

- `corpus_version`: exporter corpus version.
- `schema_version`: current schema label, currently `annotation_corpus_v1`
  (`ANNOTATION_CORPUS_SCHEMA_VERSION` in `annotation_manifest.py`).
- `created_at_utc`: export timestamp.
- `annotator_id`: CLI fallback reviewer ID.
- `include_unlabeled`: whether unlabeled rows were exported.
- `source_review_files`: input paths.
- `source_files`: input file name, suffix, size, and SHA-256 digest.
- `outputs`: named output artifact paths.
- `summary`: label counts, score distributions, reviewer counts, split summary,
  and training-readiness summary.
- `validation`: source row counts, valid row counts, reviewer IDs, warnings,
  skipped rows, duplicates, and conflicts.
- `training_readiness`: machine-readable readiness status.
- `agreement`: lightweight reviewer-overlap and binary agreement summary.
- `agreement_summary_path`: path to `agreement_summary.json`.
- `adjudication_needed_path`: path to `adjudication_needed.jsonl`.
- `reviewer_overlap_counts`: overlap count per reviewer pair.
- `unresolved_adjudication_count`: number of flagged adjudication records.
- `splits`: split strategy, seed, fractions, row counts, group counts, grouping
  field counts, output paths, and warnings.

Split metadata is also mirrored at top level for convenient shell/script access:

- `split_strategy`
- `split_seed`
- `split_fractions`
- `split_row_counts`
- `split_group_counts`
- `split_output_files`

## Training Readiness

Training readiness is separate from validation cleanliness.

Statuses:

- `ready for exploratory training`: exported labels exist and no validation
  issues were recorded.
- `usable with caution`: exported labels exist, no blocking issues were found,
  but warnings should be reviewed.
- `not ready for training`: no labels were exported, or skipped rows,
  duplicates, or conflicts require review.

A zero-label export is not training-ready even if there are no validation
errors.

## Validation Artifacts

The exporter writes all QA findings as JSONL so they can be inspected or
adjudicated outside the pipeline.

- `warnings.jsonl`: nonblocking issues such as missing reviewer metadata or
  unknown preserved fields.
- `skipped_rows.jsonl`: rows excluded from label export, including malformed
  scores or missing required IDs.
- `duplicates.jsonl`: repeated rows skipped to avoid double-counting.
- `conflicts.jsonl`: same-reviewer conflicts and cross-reviewer disagreements.

Conflicts are not automatically adjudicated. They should be reviewed before a
corpus is treated as stable training data.

## Agreement And Adjudication Artifacts

`agreement_summary.json` provides lightweight inter-reviewer QA for exported
question-data match labels. It operates on normalized labels where possible,
not raw review rows.

Current fields include:

- `schema_version`: current schema label, currently `annotation_agreement_v1`
  (`AGREEMENT_SCHEMA_VERSION` in `annotation_agreement.py`).
- `reviewer_count` and `reviewers`.
- `target_count`, `labeled_target_count`, and `multi_reviewed_target_count`.
- `reviewer_pairs`: reviewer-pair overlap, agreement, disagreement, observed
  agreement, positive agreement, negative agreement, and optional binary
  Cohen's kappa.
- `observed_agreement`: corpus-level pairwise observed agreement across
  reviewer overlaps.
- `adjudication_needed_count`: records that should be reviewed before treating
  the corpus as adjudicated.

These metrics are QA summaries only. They are not final validation statistics,
and they should not be reported as publication-grade inter-annotator agreement
without a designed annotation protocol and adjudication plan.

`adjudication_needed.jsonl` flags targets that need human review. Current
records include:

- cross-reviewer disagreements on the same match target.
- same-reviewer conflicting labels detected during validation.
- target IDs, match/question/dataset IDs, available source/document metadata,
  reviewers, labels by reviewer, source review files, and notes.

Adjudication itself is not implemented. The exporter only identifies records
that should be resolved outside the pipeline before labels are used as stable
training data.

## Human-Readable Report

`annotation_corpus_report.md` summarizes the same export in reviewer-facing
Markdown:

- source files and checksums
- label counts and distributions
- reviewer coverage
- reviewer overlap, agreement, and adjudication-needed counts
- output artifact paths
- optional split summary
- QA findings with examples
- training-readiness status and recommended next action

The report is intended for quick inspection. `manifest.json` remains the
machine-readable source of record.

## Related Future Schemas

The codebase also defines annotation schemas that are not yet fully emitted by
the current review exporter:

- `QuestionLabel`: whether a candidate is a valid open question.
- `EvidenceSpanLabel`: evidence-span supervision for question extraction.
- `ExpertPaperAnnotation`: a container for paper-level question/span labels.
- `DatasetCapability`: observed or derivable dataset capability metadata.
- `DerivedCapabilityLabel`: reviewer judgment of derived capability plausibility.

These schemas are intended for the future annotation corpus, database capability
registry, and supervised calibration layers. They should be connected through
explicit loaders/exporters before being used for model training.
