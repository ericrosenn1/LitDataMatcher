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

## Near-Term Nodes

1. **Literature ingestion node**
   - OpenAlex and PubMed queries
   - DOI and title normalization
   - citation count and concept enrichment

2. **PDF/section parsing node**
   - GROBID XML parsing with robust XML parser
   - fallback PDF text extraction
   - section-level provenance

3. **PICO/PECO extraction node**
   - population
   - exposure/intervention
   - comparator
   - outcome
   - timepoint

4. **Dataset source nodes**
   - GEO
   - SRA/ENA
   - MGnify
   - Qiita
   - ClinicalTrials.gov
   - Metabolomics Workbench

5. **Human review node**
   - expert annotation templates
   - reviewer agreement
   - adjudication exports
   - active-learning loop

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
