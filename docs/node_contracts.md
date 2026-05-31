# Node Contracts

This document defines the stable interfaces between LitDataMatcher nodes.

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
- `confidence`

## QuestionCandidate

`QuestionCandidate` is the central literature-derived object.

Key fields:

- `question_id`: deterministic stable ID
- `question`: normalized human-readable question
- `source_ids`: source document IDs
- `evidence`: supporting `Evidence` records
- `extraction_type`: explicit, future-direction, or limitation-derived
- `domain_terms`: lexical concepts used for matching
- `required_variables`: inferred data requirements
- `population`: coarse target population
- `confidence`
- `novelty_score`
- `significance_score`
- `answerability_hint`

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
- `metadata`

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
