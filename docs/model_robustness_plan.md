# Model Robustness Plan

This plan describes the steps required to move LitDataMatcher from a functional
research prototype to a high-confidence scientific discovery system.

## 1. Annotation And Evaluation Backbone

Create a versioned expert annotation corpus with:

- Explicit research questions
- Future-direction statements
- Limitation-derived questions
- Inferred open questions
- Required variables
- Population, exposure, comparator, outcome, and timepoint labels
- Dataset relevance labels for ranked question-data pairs

Recommended first milestone:

- 150 abstracts or full-text sections
- 2 annotators for 50 overlapping records
- Adjudicated gold labels
- Error taxonomy for false positives and false negatives

Primary metrics:

- Extraction precision, recall, and F1
- Calibration error for confidence scores
- Precision@k, MRR, and nDCG for ranked matches
- Expert acceptance rate of top 25 opportunities

## 2. Candidate Generation

Use conservative rule and lexical extraction to generate high-recall candidates.
Then add model-based filtering:

- Sentence-embedding similarity to discourse prototypes
- Supervised classifier trained on annotation corpus
- Optional LLM adjudication with strict JSON schema
- Section-aware priors: Introduction, Discussion, Limitations, Future Work

## 3. Ontology And Variable Harmonization

Current ontology support is lightweight. The next layer should map concepts to:

- MeSH for biomedical topics
- UMLS for clinical concepts
- EFO for experimental factors
- MONDO or DOID for diseases
- OBI for assay and measurement types
- NCBITaxon for organisms

Every mapping should preserve:

- Original text span
- Canonical concept ID
- Synonym used
- Confidence
- Source ontology version

## 4. Dataset Adapters

Production adapters should include:

- Source-specific metadata parsers
- Rate-limit handling
- Local request cache
- Raw payload preservation
- License and access extraction
- Schema drift tests
- Provenance and retrieval timestamp

Priority sources:

- OpenAlex and PubMed for literature metadata
- GEO and SRA/ENA for transcriptomic and sequencing data
- MGnify and Qiita for microbiome data
- ClinicalTrials.gov for clinical study metadata
- Metabolomics Workbench for metabolomics

## 5. Ranking Calibration

The ranking model should progress from heuristic to calibrated:

1. Hand-weighted interpretable scoring
2. Expert-labeled pairwise comparisons
3. Learning-to-rank baseline
4. Calibration curves by domain
5. Ablation study for score components

The ranking output should remain explainable even if the scoring model becomes
learned.

## 6. Publication Readiness Criteria

LitDataMatcher is publication-ready when it has:

- Reproducible code and CI
- Public annotation guidelines
- Gold benchmark corpus
- Reported extraction and ranking metrics
- At least two real live dataset adapters
- End-to-end case studies with expert validation
- Error analysis and limitations
- Archived versioned outputs
