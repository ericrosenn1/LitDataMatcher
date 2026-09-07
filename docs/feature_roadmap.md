# Feature Roadmap

## Current Completed Foundation

- Reproducible Python package and CLI
- Deterministic open-question extraction
- Ontology-backed variable normalization
- Offline curated biomedical dataset catalog
- Meta-analysis style question clustering
- Explainable question-to-dataset ranking
- Governance and feasibility assessment
- JSONL, SQLite, review CSV, and manuscript report outputs
- Evaluation and review-summary CLI commands
- Local ingestion for text, Markdown, PDF, JSONL, JATS/PMC XML, GROBID TEI XML,
  and generic XML
- Annotation export with reviewer-overlap and adjudication-needed QA artifacts
- Optional cached metadata adapters for PubMed/OpenAlex literature and
  ClinicalTrials.gov/GEO/MGnify dataset discovery
- Dataset capability export for observed and plausibly derived variables
- Daily review-queue and synthetic stress-demo CLI commands
- Exploratory ranking-threshold calibration from exported match labels
- Controlled 3-5 file manual smoke-test workflow with notes and artifact summary

## Near-Term Nodes

1. **Literature ingestion node**
   - parser-specific section provenance
   - GROBID server/client integration for direct PDF-to-TEI generation
   - PMC Open Access bulk download helpers
   - DOI and title normalization
   - citation count and concept enrichment

2. **Gold-standard annotation node**
   - larger expert-labeled question corpus
   - adjudication resolution exports
   - question-origin and evidence-span labels
   - reviewer training and calibration protocol

3. **Training/calibration node**
   - learned ranking calibration from reviewer labels
   - precision@k, MRR, nDCG, and calibration plots
   - leakage-resistant train/validation/test evaluation
   - model cards for trained components

4. **Dataset source nodes**
   - SRA/ENA
   - Qiita
   - Metabolomics Workbench
   - dbGaP/access-controlled catalog metadata
   - source-specific schema-drift tests

5. **Human review node**
   - adjudication workflow
   - lightweight local review UI
   - active-learning queue selection
   - reviewer agreement by source/document strata

## Implemented But Still Early

1. **Live metadata search**
   - PubMed and OpenAlex literature metadata
   - ClinicalTrials.gov, GEO, and MGnify dataset metadata
   - cached HTTP access
   - requires source-specific validation before publication use

2. **PDF/XML parsing node**
   - JATS/PMC XML parsing
   - GROBID TEI XML parsing
   - generic XML fallback
   - pdfminer text fallback for PDFs
   - section-level provenance remains limited

3. **Human review exports**
   - expert annotation templates
   - reviewer agreement QA
   - adjudication-needed exports
   - daily scoring queue

4. **Exploratory calibration**
   - deterministic threshold selection
   - score-bin summaries against expert labels
   - no trained predictive model yet

## Deferred Details

1. **Literature enrichment**
   - DOI and title normalization
   - citation count and concept enrichment

2. **PICO/PECO extraction node**
   - population
   - exposure/intervention
   - comparator
   - outcome
   - timepoint

## Medium-Term Nodes

- Claim/effect extraction
- Contradiction detection
- Statistical power estimation
- Study-design recommendation
- License/access/governance classifier
- Dataset-variable schema matcher
- Embedding index for literature and datasets
- API service for interactive review

## Long-Term Publication System

- Benchmark dataset release
- Manuscript figures and tables
- Reproducible case studies
- Domain-specific plug-ins
- Expert-in-the-loop ranking calibration
- Containerized deployment
