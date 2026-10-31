# Final scientific and software architecture

## Product boundary

LitDataMatcher links literature-defined research questions to existing biological data and contextualized evidence. Its primary output is a ranked, inspectable opportunity with explicit requirements, candidate datasets or dataset combinations, missing information, evidence, and a feasible next analysis. It is not merely an article summarizer, a dataset search page, or an autonomous system claiming to have performed every proposed experiment.

Use a modular Python application first. Logical workers may be functions, process pools, or queue jobs; a separate microservice and agent conversation for every transformation is unnecessary. Preserve the existing package, stable identifiers, rich JSONL/SQLite outputs, and meaningful tests where suitable. Do not retain obsolete internals solely to avoid refactoring.

## Two independently updating acquisition pipelines and a shared analytical layer

```text
Literature sources                         Dataset/reference sources
      |                                              |
search, acquire, parse                     discover, acquire metadata/files
      |                                              |
raw snapshots + source versions + hashes + retrieval records
      |                                              |
semantic extraction                       study/sample/capability profiling
      |                                              |
entity and concept resolution + normalized evidence records
      |                                              |
question discovery -> experimental requirements <-> local dataset catalog
                         |
                indexed candidate retrieval
                         |
              requirement/capability assessment
                         |
      evidence compiler <-> dated gap/novelty reassessment
                         |
     ranked opportunities + evidence bundles + analysis plans
                         |
           review UI/report + versioned evaluation
```

A new paper can invalidate a gap; a new dataset can make it answerable. Incremental changes must invalidate only the relevant downstream records and indexes. Keep scientific dependency edges as well as execution dependencies. Source synchronization, extraction, and ranking are distinct jobs, not one indivisible run.

## Representations

The information below is required; individual class names are not. Reuse and extend existing classes, introduce versioned migrations, and avoid duplicate competing schemas.

| Object | Required semantics |
|---|---|
| Document/version | Stable ID, DOI/PMID/PMCID when known, publication/version relationships, dates, source route, section structure, raw and normalized text hashes, access/license, warnings |
| Entity/mapping | Original mention, stable typed identifier candidates, organism, mapping method/resource/version, exact/synonym/broader/narrower/ortholog/related mapping type, ambiguity and provenance |
| Claim/relation | Typed subject/predicate/object, qualifiers, direction and negation, direct experiment/background/interpretation/future-work status, evidence spans, effect/unit/uncertainty when reported |
| Question | Scientific proposition, origin and evidence, scope, known facts, missing relation/measurement, dated gap status, competing explanations, importance rationale |
| Experimental requirement | Target population/system, measured exposure or intervention, comparator, outcome/estimand, assay, specimen, time/dose, unit of analysis, pairing/covariates, essential versus helpful fields |
| Study/sample/group | Repository accession and study lineage, sample-to-donor links, experimental groups/contrasts, controls, technical replicates, series/run relationships, metadata sources |
| Dataset capability | What is actually measured or derivable, feature space, metadata coverage, usable subjects/groups, design, assay, raw/processed availability, access, file/schema inspection evidence |
| Compatibility | Requirement-by-requirement result, supporting metadata locator, conflict/missingness, valid transformations, required additional work, direct/partial/indirect/not-qualified status |
| Evidence item/bundle | Proposition addressed, observation/prediction/curation type, biological conditions, source and experimental lineage, dependence groups, support/contradiction/neutral/inconclusive role, integration method |
| Match/opportunity | Question, dataset(s), capability and compatibility assessments, evidence bundle, dated unresolvedness, score components, missing needs, proposed analysis and caveats |
| Operational records | Source snapshot, run/job/attempt, artifact manifest, software/model/config versions, lineage, executed checks, cache/inference origin and failure states |

Represent unknowns explicitly with null or a typed unknown status plus reason. Zero subjects, unreported subjects, and no eligible subjects are different. Scores must reject NaN/infinity and invalid types; do not silently clamp malformed input into apparently valid evidence. Keep field-level provenance for inferred sample groups and variables, not just a citation on the dataset as a whole.

Embeddings support retrieval. They do not replace the typed records or raw text. Preserve both semantic representations and original numeric/textual observations. Use content digests for integrity and stable IDs for identity; changing a model should not change a source's identity.

## Evidence interpretation and integration

For every proposed combination, the compiler must classify both **what it says about the question** and **how it may be combined**.

Evidence roles include direct test, replication, perturbational observation, associative observation, mechanistic context, prediction, contradiction, inconclusive observation, and indirect contextual relevance. These can overlap where the data justify it. Do not turn a flat category list into a universal strength ordering. In particular, a new modality is not automatically independent evidence.

Integration modes:

1. **DIRECT_COMBINE:** measurements and identifiers can be validly joined or pooled. Require actual sample/feature alignment, compatible units/design and a documented analysis contract. No shared-subject join without shared identifiers or a justified linkage.
2. **HARMONIZED_COMBINE:** a documented transformation permits comparison or combination. Retain input values, mapping version, assumptions, discarded features, directionality and sensitivity results. Cross-species orthology or pathway mapping is not an identity assertion.
3. **EVIDENCE_SYNTHESIS:** different experiments or modalities bear on a shared proposition but cannot be pooled numerically. Keep separate contextualized evidence and dependency groups, then explain their joint relevance.
4. **CONTEXT_ONLY_OR_UNRESOLVED:** relation exists but the mapping or measured endpoint is too indirect or missing to affect the main answerability judgment. Retain it without promoting it to a direct test.
5. **NOT_COMBINABLE:** an attempted merge is invalid; preserve the rejected attempt and reason. Relevant contradictory evidence still belongs in the bundle.

All relevant evidence within configured acquired/search coverage is eligible for compilation, including inconvenient, indirect, and incompatible observations. 'Total compiler' does not require downloading every biomedical dataset or inventing a relation. Search expansion uses recorded entity/path/context constraints, a finite query budget, and explicit coverage. Do not assume complete world knowledge.

Preserve the original paper, study/cohort, sample overlap where known, and source-of-source relationships. Deduplicate exact copies and assign dependence groups for a publication, its deposited dataset, reused cohort analyses, and downstream KG copies. Unknown overlap means independence is unknown, not proven. Biolink-style primary/aggregator source fields can help resource provenance, but experimental lineage must be represented separately [S08 in file 15].

Do not assign one arbitrary vote per source or apply a universal Bayesian update to incomparable evidence. Start with explicit, interpretable assessments. Statistical pooling or posterior probabilities require a stated estimand, likelihood/dependence assumptions, and validated implementation. One small valid numeric demonstration plus explicit abstention on incompatible cases is preferable to a fake general merger.

## Question discovery and novelty

Support at least explicit unresolved/future-work extraction and one substantive cross-document gap mode, such as conflicting results, missing comparison, or an untested contextual extension. Define the missing evidence and why an available measurement could address it. An LLM-generated interesting question without source-linked rationale does not qualify.

Record statuses such as unresolved-in-searched-coverage, partly answered, answered-in-scope, contradictory, and insufficient-coverage. Retain the as-of date and sources searched. Search for later and alternative evidence before promoting novelty. Do not convert the count of publications, recency, citation count, or an author's future-work language into novelty by itself.

Compilation can decrease novelty, increase or decrease answerability, reveal a replication opportunity, or expose an unmeasured mechanism. Store the before/after reason. Already-answered questions may remain available as replication/validation opportunities but must not be ranked as novel unanswered questions.

## Dataset capability and matching

GEO series, sample, platform, SRA study/experiment/run, and ENA mirrors must resolve into appropriate levels. A study with 100,000 cells from four donors has four observed donors for donor-level inference. Count independent units separately from cells, reads, runs and technical replicates.

Profile actual sample annotations and file schemas. A title mentioning a treatment does not establish that usable treated/control groups exist. A trial registry is metadata, not automatically participant-level data. A raw read archive does not promise processed measurements or phenotype labels.

For each question produce a measurement/design contract. Use hard requirements only for genuinely indispensable scientific conditions, not minor contextual preferences. Missing indispensable metadata produces 'requires inspection', not arbitrary exclusion or confirmed compatibility. Proxy assays and mapped endpoints may support a related question if that reformulation is explicit, not silently substituted.

Candidate generation combines local lexical/identifier filters and pretrained semantic retrieval. Score only a manageable candidate set with expensive interpretation. Evaluate design fit, measured-variable coverage, independent-unit adequacy, missingness, access, analytical effort, freshness, and scientific value separately. A general sample-size threshold is not a power calculation. Where adequate effect/variance assumptions are unavailable, report adequacy unknown and explain what must be checked.

Support complementary dataset combinations as explicit plans with join/integration feasibility. Do not call two incomplete datasets jointly sufficient merely because their union of variable names looks complete; identify whether required variables are jointly observed in the needed units.

## Runtime and storage

Use immutable cached source artifacts, versioned normalized records, and disposable derived indexes. Prefer existing SQLite/JSONL; use Parquet/DuckDB and a local vector index when beneficial. Each store has one documented responsibility. Avoid unnecessarily replicating every datum in several databases or loading an entire large graph into Python objects.

Snapshot/offline mode must make no external network calls, including hidden tokenizer/model/ontology downloads. Connected mode performs bounded synchronization and reports freshness failures. Permit explicit targeted enrichment without imposing remote APIs at every matching step. Runtime-model details are in file 16.

A worker result includes input/output schema versions, source/model/config hashes, attempts, counters, evidence paths, status and next repair scope. Downstream stages may process valid partitions, but failed or missing prerequisites cannot be replaced with synthetic successes.
