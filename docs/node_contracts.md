# Node Contracts

This document defines the stable interfaces between LitDataMatcher nodes.

## SourceProvenance

`SourceProvenance` describes where a literature or dataset record came from and
how much content it contains. It is review metadata, not a claim that a source
is complete, validated, or analysis-ready.

Key fields:

- `source_type`: source family such as local text, JATS XML, PubMed, GEO, or
  MGnify
- `source_locator`: best available locator, usually URL, path, or native ID
- `content_scope`: full text, abstract-only, metadata-only, dataset metadata,
  derived capability catalog, or synthetic/demo
- `acquisition_method`: local file, API, cached API, GROBID upload, manual, or
  synthetic
- `adapter_name` / `parser_name`: implementation that produced the normalized
  record
- `retrieval_time_utc`: retrieval or local processing time when available
- `local_path` / `source_url` / `raw_record_id`: traceability fields
- `warnings` and `limitations`: caveats that should follow the record into
  review and reporting

The detailed vocabulary and adapter/parser checklists are documented in
`docs/source_ingestion_and_provenance.md`.

## Evidence

`Evidence` records where a claim came from.

Required:

- `text`

Recommended:

- `source_id`
- `title`
- `doi`
- `section`
- `sentence_index`
- `extraction_method`
- `extraction_confidence`

## QuestionCandidate

`QuestionCandidate` is the central literature-derived object.

Key fields:

- `question_id`: deterministic stable ID
- `question`: normalized human-readable question
- `source_ids`: source document IDs
- `evidence`: supporting `Evidence` records
- `question_origin`: explicit, future-direction, limitation-derived, or unspecified
- `domain_terms`: lexical concepts used for matching
- `required_variables`: inferred data requirements
- `population`: coarse target population
- `extraction_confidence`
- `novelty_score`
- `significance_score`
- `answerability`

## DatasetRecord

`DatasetRecord` normalizes repository metadata.

Key fields:

- `dataset_id`
- `title`
- `source`
- `description`
- `url`
- `variables`
- `populations`
- `organisms`
- `assay_types`
- `sample_size`
- `license`
- `access_type`
- `quality_score`
- `metadata`: optional source provenance, raw source metadata, and review caveats

## DatasetVariable

`DatasetVariable` supports variable matching.

Key fields:

- `name`
- `category`
- `observed_count`
- `completeness`
- `synonyms`

## EvidenceSynthesis

`EvidenceSynthesis` summarizes related questions.

Key fields:

- `cluster_id`
- `question_ids`
- `summary`
- `support_count`
- `contradiction_count`
- `recurrence_score`
- `evidence_strength`
- `uncertainty`

## MatchCandidate

`MatchCandidate` is a ranked opportunity.

Key fields:

- `match_id`
- `question`
- `dataset`
- `score`
- `rationale`
- `missing_variables`
- `assessments`

## MatchScore

`MatchScore` exposes the ranking calculation.

Component scores:

- `variable_overlap`
- `semantic_relevance`
- `population_fit`
- `data_quality`
- `sample_adequacy`
- `significance`
- `feasibility`
- `uncertainty_penalty`
- `governance`
- `design_fit`
- `combined`

The `combined` score should be used for sorting, but review workflows should
inspect the components and rationale before acting on a match.

## FeasibilityAssessment

`FeasibilityAssessment` explains whether a dataset can plausibly support a
question.

Key fields:

- `variable_coverage`
- `population_fit`
- `sample_adequacy`
- `longitudinal_fit`
- `assay_fit`
- `governance_reuse`
- `overall`
- `recommended_design`
- `present_variables`
- `missing_variables`
- `caveats`

## GovernanceAssessment

`GovernanceAssessment` provides reuse-risk context.

Key fields:

- `access_score`
- `license_score`
- `privacy_score`
- `reuse_score`
- `risk_flags`

## Annotation Export Labels

Completed review sheets can be converted into normalized annotation-training
artifacts with `litdatamatcher annotation-export`.

Current exported label families:

- `QuestionDataMatchLabel`: reviewer judgment of whether a dataset can answer a
  question.
- `QuestionQualityScore`: reviewer quality score for an extracted or proposed
  question.

The full artifact contract, including optional split files, validation outputs,
manifest fields, and training-readiness statuses, is documented in
`docs/annotation_export_schema.md`.
