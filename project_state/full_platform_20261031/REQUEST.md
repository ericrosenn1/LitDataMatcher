# CODEX MASTER TASK: LitDataMatcher Full Development and Integration Campaign

## Mission

Continue building and developing LitDataMatcher into a complete, production-capable, scientifically grounded biomedical research discovery platform.

**This is an implementation and development campaign, not an audit campaign.**

The objective is to deliver the full original LitDataMatcher vision through successive, functional, integrated development milestones.

Do not stop after analyzing the repository, writing recommendations, producing an implementation plan, or completing one development phase.

Implement the roadmap, run the software, test new functionality, resolve problems, integrate completed components, and proceed to the next stage.

Operate autonomously within the available Codex execution environment. When interrupted, preserve sufficient durable state for the next execution to resume without repeating completed work.

## 1. Project Context and Starting Position

LitDataMatcher identifies potentially unanswered biomedical research questions from scientific literature and finds existing public datasets that could support investigations addressing those questions.

The system must go beyond conventional literature retrieval or similarity-based dataset recommendation.

Its central contribution is connecting:

1. Biomedical literature and research findings.
2. Identification of unresolved research questions and knowledge gaps.
3. Translation of those questions into explicit experimental and analytical requirements.
4. Discovery of relevant publicly available datasets.
5. Examination of dataset structure, variables, measurements, and experimental characteristics.
6. Scientific compatibility and feasibility assessment.
7. Evidence-grounded ranking and explanation.
8. Interactive exploration and reproducible research outputs.

### Existing baseline

Use the current merged `main` branch as the starting engineering baseline.

The October 31, 2026 development assessment reported:

- Integrated modular Python research engine.
- Model-backed source-literal question extraction.
- Qwen2.5-7B extraction capabilities.
- MiniLM retrieval and adaptation.
- BM25, dense retrieval, and ranking fusion.
- Literature acquisition and source-provenance infrastructure.
- Typed experimental requirements and evidence contracts.
- Dataset candidate generation and ranking.
- Persistent checkpoints and recovery mechanisms.
- Approximately 1,317 passing integrated tests.
- React/TypeScript research workspace.
- Approximately 45.49 million historical PubMed citation events audited across 1,621 shards.
- Retrieval experiments at 50K and 100K candidate scales.
- Thirty development research cases.
- A preliminary set of 140 candidate dataset rows.

These are starting points, not final capabilities.

The previously completed assessment found several unfinished areas:

- The historical PubMed event corpus has not been fully materialized into a deduplicated current literature index.
- Scientific dataset qualification is incomplete.
- Actual biomedical dataset inspection remains limited.
- The browser application's live matching remains primarily lexical.
- The full Python model-backed research engine is not yet integrated into the interactive application.
- Independent end-to-end scientific evaluation remains incomplete.

Do not restart the project, rebuild working systems unnecessarily, or discard existing research outputs.

Preserve existing functional infrastructure and extend it.

## 2. Execution Policy: Implement, Integrate, Continue

This campaign is explicitly authorized to modify code, implement new capabilities, execute development workloads, run tests, create branches, commit changes, and integrate successfully completed work.

### Required operating behavior

1. Inspect the repository sufficiently to identify current implementations and entry points.
2. Establish the next executable development milestone.
3. Implement the missing functionality.
4. Test it against real workflows and representative inputs.
5. Fix discovered implementation defects.
6. Integrate successful work into the existing architecture.
7. Commit the completed milestone.
8. Update persistent campaign state.
9. Continue automatically to the next milestone.

Do not create a standalone repository audit, code-quality survey, readiness assessment, or speculative planning deliverable in place of development.

Testing, validation, and scientific benchmarking are integral engineering activities and should occur alongside implementation.

### Autonomous execution

Use available parallel Codex agents or worktrees for independent workstreams when supported.

Assign well-defined ownership boundaries and integrate completed changes.

Avoid simultaneous edits to the same modules unless coordinated.

If agent rotation or context replacement is necessary, persist the current milestone, completed work, blockers, and next executable actions so a successor agent can continue.

Do not treat elapsed time, agent context boundaries, or an individual completed subtask as campaign completion.

Continue through all feasible milestones in the current execution. If the execution environment terminates or requires a new invocation, leave an executable continuation mechanism and a precise resumable checkpoint.

Do not claim that an autonomous process remains active unless a real persistent process or supported scheduler has been established and verified.

## 3. Development Workstream A: Complete the Literature Infrastructure

### Goal

Transform the existing literature-acquisition infrastructure into a functional, scalable, current biomedical literature corpus supporting the full research engine.

### Implement

- Materialization of current PubMed records from historical baseline and update events.
- Correct handling of additions, replacements, updates, and deletions.
- Deduplication and PMID identity preservation.
- Efficient storage and indexing of titles, abstracts, metadata, identifiers, and provenance.
- Incremental updates without rebuilding the complete corpus.
- Searchable lexical and semantic retrieval interfaces.
- Document and passage retrieval with traceable source locations.
- Scalable processing for multi-million-record corpora.
- Resume-safe processing and recovery from interrupted batches.
- Integration with existing literature source adapters.

Reuse existing cached data and acquisition artifacts whenever valid.

Avoid unnecessary downloads or reprocessing.

Ensure the resulting corpus is directly usable by downstream question extraction and scientific discovery.

### Completion

The system can search the materialized biomedical literature corpus and retrieve source-grounded evidence with stable identifiers.

## 4. Development Workstream B: Research Question and Knowledge-Gap Discovery

### Goal

Advance the existing extraction system into a robust biomedical research-question discovery engine.

### Implement and improve

- Extraction of research questions from scientific papers.
- Identification of explicitly stated unanswered questions.
- Detection of limitations, unresolved mechanisms, and proposed future investigations.
- Structured representation of research questions.
- Separation of source-stated questions from model-inferred hypotheses.
- Evidence-linked claims and supporting passages.
- Cross-document aggregation of related questions.
- Duplicate and near-duplicate question detection.
- Question normalization, clustering, and categorization.
- Assessment of whether relevant literature already addresses the proposed question.
- Appropriate uncertainty and unsupported-claim handling.

Use existing Qwen extraction and source-fidelity infrastructure where appropriate.

Improve model prompts, structured decoding, batching, and inference efficiency when the existing implementation is insufficient.

Do not assume that larger models are necessarily better.

Compare practical approaches using actual downstream research-question utility.

### Completion

The system can process real scientific literature and produce structured, source-grounded research questions suitable for downstream dataset discovery.

## 5. Development Workstream C: Experimental Requirement Inference

### Goal

Translate biomedical research questions into scientifically meaningful dataset requirements.

### Implement

A structured requirements engine capable of representing, when relevant:

- Organism and biological system.
- Tissue, cell type, and anatomical context.
- Disease, phenotype, or biological condition.
- Experimental groups and comparison conditions.
- Perturbations, exposures, and interventions.
- Measurement modality and assay technology.
- Required genes, proteins, metabolites, or molecular features.
- Clinical outcomes and phenotype variables.
- Sample units and replication requirements.
- Temporal and longitudinal requirements.
- Cohort characteristics.
- Required metadata and covariates.
- Analytical design and feasibility constraints.
- Mandatory versus preferred requirements.
- Conditions that make a dataset unsuitable.

Use typed, machine-readable representations that can be evaluated by the downstream matching engine.

Support multiple feasible experimental designs when a research question does not imply a single unique methodology.

### Completion

The research engine can generate executable scientific requirements from extracted questions and use those requirements during dataset qualification.

## 6. Development Workstream D: Public Biomedical Dataset Discovery

### Goal

Build comprehensive, integrated discovery of real public biomedical datasets.

### Sources

Extend the current adapters to support relevant repositories, prioritizing:

- NCBI GEO.
- NCBI SRA.
- BioProject and BioSample.
- ArrayExpress / BioStudies.
- Expression Atlas.
- PRIDE / ProteomeXchange.
- Metabolomics Workbench.
- Relevant public clinical and biomedical repositories where access permits.

Use stable repository APIs and documented metadata interfaces.

Respect service limits and access restrictions.

### Required functionality

- Cross-repository search.
- Metadata normalization.
- Dataset and study identifier resolution.
- Search expansion using biological terminology.
- Synonym and ontology-aware retrieval where useful.
- Candidate deduplication.
- Provenance tracking.
- Metadata caching.
- Incremental acquisition.
- Efficient asynchronous or bounded parallel retrieval.
- Recoverable downloads and processing.
- Detection of unavailable or restricted data.

Support progressively deeper inspection:

1. Initial metadata discovery.
2. Study and experiment metadata.
3. Sample-level metadata.
4. Dataset structure and measurements.
5. Detailed data inspection where needed.

Avoid downloading entire large datasets when repository metadata or smaller inspection artifacts can establish suitability.

### Completion

The research engine can discover relevant real datasets across supported repositories and provide normalized candidate records to the scientific matching engine.

## 7. Development Workstream E: Scientific Dataset Compatibility Engine

### Goal

Implement the defining capability of LitDataMatcher: determining whether a discovered dataset can support investigation of a specific unanswered research question.

This is not merely text similarity.

### Implement

- Question-to-dataset requirement matching.
- Biological-system compatibility.
- Assay and measurement compatibility.
- Experimental condition matching.
- Sample-group and comparator matching.
- Phenotype and outcome availability.
- Metadata completeness assessment.
- Data accessibility and reusability checks.
- Hard constraints and soft preferences.
- Contradiction detection.
- Explicit missing-requirement identification.
- Feasibility classification.
- Confidence and evidence representation.
- Ranking of scientifically compatible candidates.
- Explanations for rejected or uncertain candidates.

Use real dataset contents where necessary to resolve critical questions about feasibility.

Integrate existing dataset inspectors and extend support for relevant biomedical formats.

For each match, produce a structured explanation connecting:

**Research question → required experiment/data → observed dataset characteristics → compatibility decision → supporting evidence.**

### Initial development milestone

Use the existing 30 development cases and 140 candidate rows as seed material for building and exercising the matching engine.

Target at least 12 scientifically defensible positive matches as an initial functional demonstration.

This is an implementation milestone, not a separate audit project.

If existing candidate rows are unsuitable, improve retrieval and add real development examples rather than forcing positive results.

Continue expanding supported study types and scientific requirements after the initial milestone.

### Completion

The system produces scientifically interpretable, source-grounded compatibility decisions on actual public biomedical datasets.

## 8. Development Workstream F: Retrieval, Ranking, and Model Improvements

### Goal

Optimize the full question-to-dataset discovery process.

Use the existing BM25, MiniLM, dense retrieval, and rank-fusion implementations as foundations.

Develop and compare:

- Lexical retrieval.
- Dense semantic retrieval.
- Hybrid retrieval and rank fusion.
- Requirement-aware reranking.
- Ontology-assisted retrieval.
- Structured scientific compatibility scoring.
- Model-assisted reranking where beneficial.
- Multi-stage candidate generation and qualification.

The previous development benchmark found BM25 stronger than the fine-tuned MiniLM model on source-paper retrieval.

Do not assume that improving paper retrieval automatically improves dataset matching.

Optimize against the actual end-to-end scientific discovery objective.

Track:

- Candidate recall.
- Precision@k.
- Recall@k.
- Ranking quality.
- Scientific feasibility classification.
- Incompatible-dataset rejection.
- Appropriate no-match decisions.
- Runtime and resource consumption.

Compare computational costs and performance.

Preserve reproducible model versions, configurations, and benchmark outputs.

### Completion

The integrated retrieval and ranking system demonstrates measurable advantages where justified by the scientific task, while retaining simple baselines as comparators.

## 9. Development Workstream G: Integrate the Research Engine With the Web Application

### Goal

Turn the current React/TypeScript research workspace into a functional interface for the complete Python research engine.

### Implement

A stable backend integration layer supporting:

- Literature search.
- Research-question extraction.
- Question inspection and selection.
- Experimental requirement generation.
- Dataset discovery.
- Scientific compatibility assessment.
- Candidate ranking and filtering.
- Evidence inspection.
- Research dossier generation.
- Export of structured results.
- Long-running job status.
- Resume and recovery for interrupted jobs.

The existing browser-only lexical matcher may remain available as a lightweight mode.

However, it must be clearly distinguished from the full scientific matching workflow.

### User experience

Develop a coherent research workflow in which a user can:

1. Search biomedical literature or submit a scientific question.
2. Inspect extracted unanswered questions.
3. Select or modify a question.
4. View inferred experimental requirements.
5. Launch public dataset discovery.
6. Examine ranked candidate datasets.
7. Inspect experimental compatibility and evidence.
8. Generate a reproducible research opportunity dossier.

Expose useful progress and execution details without requiring the user to understand internal pipeline architecture.

Avoid unnecessary visual complexity.

Prioritize scientific functionality, responsive interaction, and efficient navigation.

### Completion

A user can run the full research-question-to-dataset workflow through the application without manually orchestrating separate Python scripts.

## 10. Development Workstream H: Scale, Automation, and Campaign Execution

### Goal

Make LitDataMatcher capable of sustained, large-scale research discovery workloads.

### Implement

- Persistent job and campaign state.
- Incremental processing.
- Idempotent stages.
- Checkpoint-based recovery.
- Bounded retries.
- Queue management.
- Parallel execution where beneficial.
- CPU, GPU, RAM, and disk resource controls.
- Model batching.
- Cache reuse.
- Failure isolation.
- Structured logs.
- Progress and throughput metrics.
- Automatic continuation between completed stages.
- Clean termination and resume behavior.

Prefer extending the existing campaign-controller and checkpoint infrastructure rather than introducing a second competing system.

### Large-scale capabilities

Support progressive workload sizes, such as:

- Small development samples.
- Thousands of documents.
- Tens of thousands of documents.
- Hundreds of thousands of documents.
- Larger literature-scale campaigns where computationally practical.

Make scale configurable rather than hard-coded.

The user should be able to launch a research campaign, monitor progress, pause execution, resume execution, and recover successful intermediate results.

### Completion

The platform can execute sustained multi-stage discovery campaigns without requiring manual intervention between normal processing stages.

## 11. Development Workstream I: Scientific Evidence, Results, and Reproducibility

### Goal

Make every research opportunity produced by the system inspectable and reproducible.

### Implement

- Stable result identifiers.
- Original literature citations.
- Dataset accession references.
- Source passages.
- Explicit scientific requirements.
- Compatibility evidence.
- Ranking explanations.
- Processing timestamps.
- Pipeline configuration.
- Model and software versions.
- Exportable machine-readable records.
- Human-readable research dossiers.
- Reproducible campaign manifests.

Provide useful outputs in existing supported formats, including JSON/JSONL, CSV, SQLite, and structured reports where appropriate.

Integrate scientific evaluation into normal development.

Create and maintain development fixtures covering positive matches, incompatible datasets, ambiguous cases, and no-fit questions.

Preserve a genuinely untouched final evaluation set until the associated evaluation protocol is finalized.

Do not consume the previously reserved GSE279879 final holdout during iterative development.

### Completion

The platform's outputs can be inspected, reproduced, compared, and used in a scientific methods manuscript.

## 12. Development Workstream J: Publication-Oriented Scientific Evaluation

### Goal

Complete the development and research evidence needed to support a peer-reviewed methods publication.

This is a downstream stage of the development campaign, not a reason to interrupt implementation for a separate audit.

### Develop

- Reproducible scientific benchmark datasets.
- Independently reviewed reference cases.
- Relevant conventional retrieval baselines.
- Ablation experiments.
- End-to-end matching evaluations.
- Dataset feasibility error analysis.
- Performance comparisons.
- Scalability benchmarks.
- Reproducible figures and results.
- A documented experimental protocol.

The scientific contribution should focus on identifying unanswered biomedical questions and matching them to experimentally suitable public datasets.

Do not frame conventional paper retrieval alone as the project's principal contribution.

Once evaluation is complete, generate publication-oriented results and documentation.

Do not fabricate scientific outcomes or label unverified candidate matches as established results.

## 13. Workstream Coordination and Parallel Development

Where the environment supports parallel agents or isolated worktrees, organize the campaign into coordinated workstreams.

Suggested assignments:

**Agent A: Literature and corpus infrastructure**
- PubMed materialization.
- Indexing.
- Incremental updates.
- Search integration.

**Agent B: Scientific extraction and requirements**
- Question extraction.
- Knowledge-gap detection.
- Typed experimental requirements.

**Agent C: Dataset acquisition and inspection**
- Repository adapters.
- Dataset discovery.
- Metadata normalization.
- Real dataset inspection.

**Agent D: Matching and model research**
- Scientific compatibility engine.
- Retrieval improvements.
- Ranking and evaluation.

**Agent E: Application integration**
- Backend API.
- Frontend workflows.
- Job progress and research dossiers.

**Agent F: Integration and campaign orchestration**
- Shared contracts.
- Cross-component integration.
- Persistent controller.
- Testing.
- Milestone integration.

Parallelize only genuinely independent work.

Coordinate interfaces before merging dependent implementations.

If parallel agents are unavailable, execute the workstreams sequentially according to dependencies.

## 14. Dependency-Aware Execution Order

Do not require every workstream to finish before another can begin.

Use staged integration.

### Phase 1: Operational foundation

- Confirm the merged engineering baseline.
- Establish durable campaign state.
- Identify reusable models, data, and controllers.
- Prepare isolated development workstreams.
- Repair immediate blockers preventing execution.

### Phase 2: Core scientific discovery

- Improve question extraction.
- Implement and extend experimental requirements.
- Expand biomedical dataset acquisition.
- Implement real dataset inspection.
- Build scientific compatibility and qualification.

### Phase 3: Complete end-to-end execution

- Connect question extraction to dataset discovery.
- Connect dataset inspection to compatibility assessment.
- Implement final ranking and evidence dossiers.
- Demonstrate real question-to-dataset workflows.

### Phase 4: Corpus and scale

- Complete PubMed corpus materialization.
- Integrate large-scale indexing and retrieval.
- Improve batching, caching, resource controls, and campaign recovery.
- Test progressively larger workloads.

### Phase 5: Full application integration

- Implement backend interfaces.
- Connect React workflows to the research engine.
- Add job progress, evidence exploration, and result exports.
- Verify complete user-facing execution.

### Phase 6: Optimization and scientific evaluation

- Improve full-pipeline performance.
- Compare baselines and models.
- Extend scientific evaluation cases.
- Evaluate ranking, feasibility, abstention, and scalability.
- Conduct final evaluation only after the protocol is ready.

### Phase 7: Research-ready release

- Consolidate completed features.
- Update documentation.
- Produce reproducible demonstration workflows.
- Prepare results for a methods manuscript.
- Complete integration and release checks.

Progress through these phases automatically.

Reorder independent implementation work when doing so improves throughput without breaking dependencies.

## 15. Persistent Campaign State and Recovery

Create or extend a durable development-campaign state mechanism using the repository's existing conventions.

It must record:

- Campaign identifier.
- Current phase.
- Active workstreams.
- Milestone status.
- Completed changes.
- Associated commits.
- Tests executed.
- Successful data products.
- Unfinished tasks.
- Recoverable failures.
- Blocking dependencies.
- Next executable actions.

Use atomic state writes and existing checkpoint conventions.

Maintain a readable campaign summary.

Support a resume command that:

1. Loads the previous campaign state.
2. Verifies relevant outputs and commits.
3. Preserves successful completed work.
4. Requeues unfinished or failed work.
5. Continues with the next eligible milestone.

The campaign must not repeatedly restart from Phase 1 after an interruption.

If a long-running controller can remain active in the actual environment, implement and verify it.

Otherwise, leave a reliable resume entry point and continuation instructions for the next Codex session.

Do not simulate persistence with status files that no real execution process consumes.

## 16. Repository and Integration Rules

- Work from the latest intended `main` baseline.
- Preserve existing public interfaces where practical.
- Prefer incremental modifications over wholesale replacement.
- Reuse existing code, tests, caches, models, and pipeline infrastructure.
- Avoid duplicated controllers and overlapping subsystems.
- Keep implementation modules reasonably separated.
- Add dependencies only when justified.
- Commit coherent, functional milestones.
- Integrate completed branches or worktrees.
- Run relevant tests after each milestone.
- Run broader integration tests at suitable checkpoints.
- Resolve regressions introduced by the campaign.
- Preserve reproducible commands and configuration.
- Do not overwrite unrelated user work.
- Do not force-push or destructively rewrite shared history.

Do not publish to PyPI or Zenodo.

Do not initiate unrelated packaging, badge, or cosmetic documentation campaigns.

Documentation updates should accompany real implementation changes.

If remote merge or deployment requires permissions unavailable in the execution environment, preserve the completed branch and provide the precise continuation action rather than claiming completion.

## 17. Resource Management

Use available compute resources efficiently.

Detect available CPUs, RAM, GPUs, and storage.

Do not assume that a GPU is available in every Codex execution environment.

Prefer existing model assets and caches.

Apply bounded concurrency and resource-aware batching.

Avoid redundant literature downloads, repeated model training, or unnecessary rebuilding of previously completed outputs.

For substantial computational campaigns, expose configurable resource limits and use a conservative default profile.

Do not silently launch unbounded multi-day workloads.

Preserve successful intermediate results and resume unfinished work.

## 18. Required Reporting

Provide concise progress reports at meaningful milestones.

Report:

- Implemented functionality.
- Actual working end-to-end capabilities.
- Tests and executed examples.
- Real scientific results where available.
- Remaining developmental limitations.
- Commits and branches.
- Next implementation milestone.

Distinguish implemented, tested, demonstrated, and scientifically evaluated capabilities.

Do not substitute lengthy assessments for working code.

Provide a final campaign summary when all currently executable milestones are finished or when further execution is genuinely blocked.

A blocked optional workstream should not prevent progress on independent workstreams.

## 19. Success Criteria

The campaign's target outcome is a LitDataMatcher application capable of:

1. Searching a large biomedical literature corpus.
2. Identifying source-grounded unanswered research questions.
3. Inferring explicit experimental requirements.
4. Searching multiple public biomedical data repositories.
5. Inspecting real dataset structures and sample metadata.
6. Determining scientific compatibility with research questions.
7. Ranking feasible dataset candidates.
8. Explaining matches and incompatibilities using verifiable evidence.
9. Producing reproducible research opportunity dossiers.
10. Running the complete workflow through an integrated application.
11. Executing substantial automated discovery campaigns with recovery.
12. Supporting rigorous scientific benchmarking and publication.

Completion requires functional integration, not merely the presence of code implementing individual stages.

## 20. Immediate Instruction

**Begin implementation now.**

Start from the existing repository state, determine the next actionable dependency, and implement it.

Do not stop after repository inspection.

Do not create a new standalone audit campaign.

Do not return only a plan.

Do not restrict development to the previously examined 30 cases.

Use those cases as existing development resources while expanding the system toward the complete scientific discovery vision.

Proceed through the workstreams, integrating and committing successful milestones, maintaining durable checkpoint state, and executing the next available development task until the campaign reaches its full target or encounters a genuine external blocker.

The governing priority is:

**BUILD → EXECUTE → TEST → FIX → INTEGRATE → COMMIT → CONTINUE.**

Deliver a functioning, integrated LitDataMatcher scientific discovery platform.