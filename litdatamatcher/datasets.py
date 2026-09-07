"""Dataset discovery and classification node."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Protocol

from .provenance import curated_catalog_provenance
from .schemas import DatasetRecord, DatasetVariable, QuestionCandidate, stable_id
from .text import extract_domain_terms, infer_population, infer_required_variables, lexical_similarity


class DataSourceAdapter(Protocol):
    """Interface implemented by dataset catalog adapters."""

    name: str

    def search(self, query: str) -> list[DatasetRecord]:
        """Return candidate dataset records for a query."""


def _var(name: str, category: str, count: int, completeness: float, *synonyms: str) -> DatasetVariable:
    """Build a normalized variable record for the curated catalog."""

    return DatasetVariable(
        name=name,
        category=category,
        observed_count=count,
        completeness=completeness,
        synonyms=list(synonyms),
    )


def _cap(name: str, category: str, count: int, completeness: float, *synonyms: str) -> DatasetVariable:
    """Build a study-design or ML-readiness capability variable."""

    return _var(name, category, count, completeness, *synonyms)


def _capability_annotations(*entries: dict[str, object]) -> dict[str, object]:
    """Return conservative curated capability metadata for review surfaces."""

    return {
        "capability_annotations": list(entries),
        "capability_caveats": [
            "Curated capability metadata is a review aid, not evidence of raw data access or analysis readiness.",
            "Study-specific source inspection is required before treating proxy capabilities as answerable variables.",
        ],
    }


class CuratedBiomedicalCatalogAdapter:
    """Offline biomedical catalog used for deterministic demos and tests.

    The records are intentionally conservative summaries of common public data
    sources. They allow the pipeline to be functional and reproducible without
    requiring API keys or network access, while preserving the same schema used
    by future live repository adapters.
    """

    name = "curated_biomedical_catalog"

    def __init__(self) -> None:
        # Curated records are deterministic stand-ins for future live database adapters.
        self.records = [
            DatasetRecord(
                dataset_id="qiita_microbiome_antibiotics_longitudinal",
                title="Qiita longitudinal human gut microbiome antibiotic exposure studies",
                source="Qiita",
                url="https://qiita.ucsd.edu/",
                description=(
                    "Curated microbiome studies with 16S and metadata fields relevant to "
                    "antibiotic exposure, body site, age, and longitudinal sampling."
                ),
                variables=[
                    _var("microbiome_composition", "omics", 1500, 0.9, "16s", "microbiota"),
                    _var("antibiotic_exposure", "exposure", 900, 0.72, "antibiotics"),
                    _var("body_site", "metadata", 1500, 0.95, "sample_site"),
                    _var("age", "metadata", 1200, 0.82),
                    _var("timepoint", "temporal_structure", 850, 0.7, "visit"),
                    _cap("sample_size", "study_design_feature", 2000, 0.95, "cohort_size"),
                    _cap("longitudinal_time", "temporal_structure", 850, 0.7, "timepoint"),
                    _cap("predictor", "derived_or_proxy_capability", 1500, 0.55, "feature", "16s_feature"),
                    _cap("covariate", "metadata_variable", 1200, 0.65, "age", "body_site"),
                ],
                populations=["human", "adult", "pediatric"],
                organisms=["human-associated microbiome"],
                assay_types=["16S rRNA sequencing"],
                sample_size=2000,
                license="varies by study",
                access_type="public metadata with study-specific access",
                quality_score=0.78,
                metadata=_capability_annotations(
                    {
                        "capability": "sample_size",
                        "capability_type": "study_design_feature",
                        "support": "direct_metadata",
                        "evidence": "Curated record carries study-level sample_size.",
                    },
                    {
                        "capability": "longitudinal_time",
                        "capability_type": "temporal_structure",
                        "support": "direct_metadata",
                        "evidence": "Description and variables indicate longitudinal sampling/timepoint metadata.",
                    },
                    {
                        "capability": "predictor",
                        "capability_type": "derived_or_proxy_capability",
                        "support": "proxy",
                        "evidence": "Microbiome profiles may serve as features after source-specific processing.",
                    },
                ),
            ),
            DatasetRecord(
                dataset_id="mgnify_human_gut_metagenomics",
                title="MGnify human gut metagenomic assemblies and taxonomic profiles",
                source="MGnify",
                url="https://www.ebi.ac.uk/metagenomics/",
                description=(
                    "Public metagenomic profiles across human gut studies with taxonomic "
                    "composition, functional annotations, body site, and study metadata."
                ),
                variables=[
                    _var("microbiome_composition", "omics", 5000, 0.9, "metagenomics"),
                    _var("body_site", "metadata", 4800, 0.85),
                    _var("disease_activity", "phenotype", 900, 0.45, "IBD"),
                    _var("age", "metadata", 2500, 0.55),
                    _cap("sample_size", "study_design_feature", 5000, 0.95, "cohort_size"),
                    _cap("predictor", "derived_or_proxy_capability", 5000, 0.6, "feature", "metagenomic_feature"),
                    _cap("covariate", "metadata_variable", 2500, 0.45, "age", "body_site"),
                ],
                populations=["human", "adult"],
                organisms=["human gut microbiome"],
                assay_types=["shotgun metagenomics"],
                sample_size=5000,
                license="open metadata",
                access_type="public",
                quality_score=0.82,
                metadata=_capability_annotations(
                    {
                        "capability": "sample_size",
                        "capability_type": "study_design_feature",
                        "support": "direct_metadata",
                        "evidence": "Curated record carries study-level sample_size.",
                    },
                    {
                        "capability": "predictor",
                        "capability_type": "derived_or_proxy_capability",
                        "support": "proxy",
                        "evidence": "Metagenomic profiles may serve as model features after study-specific processing.",
                    },
                ),
            ),
            DatasetRecord(
                dataset_id="geo_ibd_transcriptomics",
                title="GEO inflammatory bowel disease host transcriptomics cohorts",
                source="GEO",
                url="https://www.ncbi.nlm.nih.gov/geo/",
                description=(
                    "Host transcriptomic datasets from intestinal tissue and blood in IBD, "
                    "including disease state, treatment response, and inflammation markers."
                ),
                variables=[
                    _var("transcriptomics", "omics", 1800, 0.92, "rna-seq", "gene_expression"),
                    _var("disease_activity", "phenotype", 1600, 0.78, "inflammation"),
                    _var("treatment", "clinical", 900, 0.62, "therapy"),
                    _var("outcome", "clinical", 850, 0.58, "response"),
                    _cap("sample_size", "study_design_feature", 1800, 0.95, "cohort_size"),
                    _cap("predictor", "derived_or_proxy_capability", 1800, 0.68, "feature", "gene_expression_feature"),
                    _cap("class_label", "supervised_learning_label", 1600, 0.55, "label", "outcome_label", "disease_state"),
                    _cap("outcome_label", "evaluation_outcome", 850, 0.58, "label", "response"),
                    _cap("covariate", "metadata_variable", 900, 0.5, "treatment", "disease_activity"),
                ],
                populations=["human", "adult", "pediatric"],
                organisms=["human"],
                assay_types=["RNA-seq", "microarray"],
                sample_size=1800,
                license="public repository terms",
                access_type="public",
                quality_score=0.8,
                metadata=_capability_annotations(
                    {
                        "capability": "predictor",
                        "capability_type": "derived_or_proxy_capability",
                        "support": "proxy",
                        "evidence": "Transcriptomic measurements may serve as features after source-specific normalization.",
                    },
                    {
                        "capability": "class_label",
                        "capability_type": "supervised_learning_label",
                        "support": "partial_metadata",
                        "evidence": "Disease state and treatment response labels are represented in curated metadata.",
                    },
                    {
                        "capability": "outcome_label",
                        "capability_type": "evaluation_outcome",
                        "support": "partial_metadata",
                        "evidence": "Outcome/response metadata is curated but must be checked per GEO series.",
                    },
                ),
            ),
            DatasetRecord(
                dataset_id="clinicaltrials_ibd_interventions",
                title="ClinicalTrials.gov IBD intervention and outcome registry records",
                source="ClinicalTrials.gov",
                url="https://clinicaltrials.gov/",
                description=(
                    "Trial registry records with interventions, eligibility criteria, outcomes, "
                    "sample sizes, and status for IBD and microbiome-related studies."
                ),
                variables=[
                    _var("treatment", "clinical", 1200, 0.95, "intervention"),
                    _var("outcome", "clinical", 1200, 0.9, "endpoint"),
                    _var("age", "metadata", 1200, 0.85),
                    _var("sex", "metadata", 900, 0.65),
                    _var("disease_activity", "phenotype", 1000, 0.7),
                    _cap("sample_size", "study_design_feature", 1200, 0.95, "cohort_size"),
                    _cap("outcome_label", "evaluation_outcome", 1200, 0.9, "label", "endpoint"),
                    _cap("class_label", "supervised_learning_label", 1200, 0.45, "label", "study_arm", "outcome_label"),
                    _cap("covariate", "metadata_variable", 1200, 0.72, "eligibility", "age", "sex"),
                ],
                populations=["human", "adult", "pediatric"],
                organisms=["human"],
                assay_types=["clinical registry"],
                sample_size=1200,
                license="public domain US government work",
                access_type="public",
                quality_score=0.74,
                metadata=_capability_annotations(
                    {
                        "capability": "outcome_label",
                        "capability_type": "evaluation_outcome",
                        "support": "direct_registry_metadata",
                        "evidence": "Registry records include outcome/end point definitions.",
                    },
                    {
                        "capability": "class_label",
                        "capability_type": "supervised_learning_label",
                        "support": "partial_metadata",
                        "evidence": "Study arms/outcomes can define labels, but patient-level analysis data are not implied.",
                    },
                ),
            ),
            DatasetRecord(
                dataset_id="metabolomics_workbench_gut_inflammation",
                title="Metabolomics Workbench gut inflammation and diet studies",
                source="Metabolomics Workbench",
                url="https://www.metabolomicsworkbench.org/",
                description=(
                    "Metabolomics studies with diet, inflammation, treatment, and biospecimen "
                    "metadata that can complement microbiome or host omics analyses."
                ),
                variables=[
                    _var("metabolomics", "omics", 700, 0.86, "metabolites"),
                    _var("diet", "exposure", 300, 0.55, "nutrition"),
                    _var("disease_activity", "phenotype", 450, 0.58),
                    _var("treatment", "clinical", 250, 0.42),
                    _cap("sample_size", "study_design_feature", 700, 0.9, "cohort_size"),
                    _cap("predictor", "derived_or_proxy_capability", 700, 0.62, "feature", "metabolomic_feature"),
                    _cap("outcome_label", "evaluation_outcome", 450, 0.45, "label", "disease_activity"),
                    _cap("covariate", "metadata_variable", 300, 0.48, "diet", "treatment"),
                ],
                populations=["human", "mouse"],
                organisms=["human", "mouse"],
                assay_types=["LC-MS", "GC-MS"],
                sample_size=700,
                license="varies by study",
                access_type="public metadata with study-specific files",
                quality_score=0.72,
                metadata=_capability_annotations(
                    {
                        "capability": "predictor",
                        "capability_type": "derived_or_proxy_capability",
                        "support": "proxy",
                        "evidence": "Metabolite abundances may serve as model features after source-specific processing.",
                    },
                    {
                        "capability": "outcome_label",
                        "capability_type": "evaluation_outcome",
                        "support": "partial_metadata",
                        "evidence": "Disease activity metadata may support labels in some studies.",
                    },
                ),
            ),
        ]
        for record in self.records:
            _attach_curated_catalog_provenance(record)

    def search(self, query: str) -> list[DatasetRecord]:
        """Return catalog records with lexical or variable overlap."""

        query_terms = set(extract_domain_terms(query, max_terms=20))
        query_vars = set(infer_required_variables(query))
        scored: list[tuple[float, DatasetRecord]] = []
        for record in self.records:
            # Blend text, inferred-variable, and domain-term signals for transparent ranking.
            text_score = lexical_similarity(query, record.searchable_text())
            var_score = len(query_vars & record.variable_aliases()) / max(1, len(query_vars))
            term_score = len(query_terms & set(extract_domain_terms(record.searchable_text(), 20))) / max(
                1, len(query_terms)
            )
            score = max(text_score, 0.7 * var_score + 0.3 * term_score)
            if score > 0 or not query_terms:
                scored.append((score, record))
        return [record for _, record in sorted(scored, key=lambda item: item[0], reverse=True)]


class JsonlCatalogAdapter:
    """Adapter for user-supplied JSONL dataset catalogs."""

    name = "jsonl_catalog"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.records = load_dataset_catalog(self.path)

    def search(self, query: str) -> list[DatasetRecord]:
        """Search local JSONL records by lexical similarity."""

        scored = [(lexical_similarity(query, record.searchable_text()), record) for record in self.records]
        return [record for score, record in sorted(scored, key=lambda item: item[0], reverse=True) if score > 0]


def classify_dataset_record(raw: dict) -> DatasetRecord:
    """Normalize a raw repository metadata dictionary into a DatasetRecord."""

    title = raw.get("title") or raw.get("name") or raw.get("accession") or "Untitled dataset"
    description = raw.get("description") or raw.get("summary") or raw.get("abstract") or ""
    source = raw.get("source") or raw.get("repository") or "unknown"
    variables = raw.get("variables")
    if not variables:
        # Sparse external metadata still becomes usable by inferring a minimal variable list.
        variables = [
            {"name": variable, "category": "inferred", "observed_count": 0, "completeness": 0.5}
            for variable in infer_required_variables(f"{title} {description}")
        ]
    dataset_id = raw.get("dataset_id") or raw.get("id") or stable_id("dataset", source, title)
    return DatasetRecord(
        dataset_id=dataset_id,
        title=title,
        source=source,
        description=description,
        url=raw.get("url", ""),
        variables=[DatasetVariable.from_dict(v) if isinstance(v, dict) else DatasetVariable(v) for v in variables],
        populations=raw.get("populations") or [infer_population(f"{title} {description}")],
        organisms=raw.get("organisms", []),
        assay_types=raw.get("assay_types", []),
        sample_size=raw.get("sample_size", 0),
        license=raw.get("license", "unknown"),
        access_type=raw.get("access_type", "unknown"),
        quality_score=raw.get("quality_score", 0.5),
        metadata=raw.get("metadata", {}),
    )


def _attach_curated_catalog_provenance(record: DatasetRecord) -> None:
    """Attach advisory provenance to built-in offline catalog records."""

    metadata = dict(record.metadata or {})
    metadata.setdefault(
        "source_provenance",
        curated_catalog_provenance(
            dataset_id=record.dataset_id,
            source_name=record.source,
            source_url=record.url,
        ).to_dict(),
    )
    record.metadata = metadata


def load_dataset_catalog(path: str | Path) -> list[DatasetRecord]:
    """Load a JSONL dataset catalog into validated records."""

    records: list[DatasetRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                # Row-specific errors make catalog curation easier to debug.
                records.append(classify_dataset_record(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid dataset catalog row {line_number}: {exc}") from exc
    return records


def default_adapters(catalog_path: str | Path | None = None) -> list[DataSourceAdapter]:
    """Return configured data-source adapters."""

    adapters: list[DataSourceAdapter] = [CuratedBiomedicalCatalogAdapter()]
    if catalog_path:
        # User catalogs take precedence while the curated catalog remains a fallback.
        adapters.insert(0, JsonlCatalogAdapter(catalog_path))
    return adapters


def discover_datasets_for_question(
    question: QuestionCandidate,
    adapters: Iterable[DataSourceAdapter] | None = None,
    top_n: int = 10,
) -> list[DatasetRecord]:
    """Search all adapters for datasets relevant to a normalized question."""

    adapters = list(adapters or default_adapters())
    query = " ".join(
        [
            question.question,
            " ".join(question.domain_terms),
            " ".join(question.required_variables),
            question.population,
        ]
    )
    seen: set[str] = set()
    records: list[DatasetRecord] = []
    for adapter in adapters:
        for record in adapter.search(query):
            # Adapter outputs are deduplicated by stable dataset IDs before ranking.
            if record.dataset_id not in seen:
                seen.add(record.dataset_id)
                records.append(record)
    return records[:top_n]


def discover_datasets_for_topic(topic: str, top_n: int = 5) -> list[DatasetRecord]:
    """Convenience search used by the streaming data worker."""

    pseudo_question = QuestionCandidate(
        question_id=stable_id("question", topic),
        question=f"What available data can address {topic}?",
        domain_terms=extract_domain_terms(topic),
        required_variables=infer_required_variables(topic),
        population=infer_population(topic),
    )
    return discover_datasets_for_question(pseudo_question, top_n=top_n)


def summarize_dataset_matches(topic: str, top_n: int = 5) -> dict:
    """Return a compact worker payload for a topic-level data search."""

    records = discover_datasets_for_topic(topic, top_n=top_n)
    variable_counts: dict[str, int] = {}
    total_samples = 0
    quality_scores: list[float] = []
    for record in records:
        # The worker summary is intentionally compact for streaming/status displays.
        total_samples += record.sample_size
        quality_scores.append(record.quality_score)
        for variable in record.variables:
            variable_counts[variable.normalized_name] = (
                variable_counts.get(variable.normalized_name, 0) + variable.observed_count
            )
    feasibility = 0.0
    if records:
        avg_quality = sum(quality_scores) / len(quality_scores)
        coverage = min(1.0, len(variable_counts) / 8.0)
        sample_score = min(1.0, total_samples / 5000.0)
        feasibility = round(0.45 * avg_quality + 0.35 * coverage + 0.2 * sample_score, 3)
    return {
        "topic": topic,
        "datasets": [record.dataset_id for record in records],
        "variables": variable_counts,
        "samples": total_samples,
        "feasibility_score": feasibility,
        "sources": [record.source for record in records],
    }
