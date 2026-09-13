# LitDataMatcher 0.3.0: release candidate, use and remaining validation

Version 0.3.0 is the final-campaign delivery target. **Final release acceptance is pending.** This document records implemented scope and verified component results; it is not a clean-install receipt, final scientific-case result, independent approval or publication claim. The current 74-row [requirements matrix](final_audit/FINAL_REQUIREMENTS_MATRIX.md) identifies the remaining gates.

## What changed

The application now connects qualified repository metadata to explicit experimental requirements, preserves unknown metadata and biological independence, and produces source-linked evidence bundles and review dossiers. Recent repairs address identifier/lifecycle reconciliation, source-specific correction/retraction notices, per-response-page provenance, offline cache isolation, atomic cache writes, strict adjudication and unsupported calibration claims. Controlled organism aliases were added only after source qualification and a reproduced real benchmark failure.

The expanded acquisition contains 1,600 unique literature identifiers and 487 catalog accessions across cancer, environmental microbiome, metabolic and neurologic contexts. A separate qualification adds 65 proteomics/metabolomics accessions. The combined 552 catalog identifiers comprise 395 ClinicalTrials.gov, 75 MGnify, 17 ENA/SRA, 25 PRIDE and 40 Metabolomics Workbench records. These are source identifiers, not 552 proven independent cohorts. The literature sample contains 1,456 abstracts and 144 metadata-only records. It is not a new 1,600-full-text corpus.

All 13 expanded acquisition partitions replayed offline, and the omics supplement passed its independent normalized replay. The real CPU scale benchmark passed at 100/32, 500/100 and 1,600/552 literature/catalog sizes. At the full size, warmed FTS p95 was 21.65 ms, matching took 0.0523 seconds per 1,000 actual candidate assessments, and sampled peak process-tree RSS was 1.162 GiB. A child committed 276 catalog records and exited nonzero; a different child resumed only the remaining 276, with exact IDs/payloads and no duplicate versions. These measurements concern the declared local metadata workload. The original failure and the repaired run remain preserved.

The benchmark's 17 source-derived metadata questions contain genuine observed fits, wrong-organism/wrong-modality cases and unjudged records. Compatibility-aware methods produced zero confirmed-invalid top results; lexical and MiniLM retrieval produced two and three respectively. The questions derive from the same catalog, and the reference vocabulary is bounded: this is not held-out biological answerability or calibrated scientific accuracy. See the exact receipts and limitations in [the benchmark report](final_audit/PHASE2_BENCHMARK_RESULT.md).

## Qualified scope and interpretation

| Area | Supported scope | Limit that remains visible |
| --- | --- | --- |
| Literature | Local files, bounded biomedical metadata/full-text routes, source snapshots, identity/lifecycle handling and cache replay | Access failure, missing text, malformed source data and partial search coverage do not establish absent evidence or novelty |
| Catalogs | GEO-oriented historical validation plus selected clinical, sequencing, microbiome, proteomic and metabolomic metadata routes | Registry entries are not participant-level measurements; technical runs, cells, samples and accessions are not interchangeable biological units |
| Compatibility | Explicit observed requirements, qualified organism/assay mappings, feature/unit/design contracts and hard eligibility | Similarity cannot repair a known essential mismatch; missing or ambiguous source metadata needs inspection |
| Evidence | Source spans, exact lineage/dependence, contradictions, indirect/orthogonal context and provenance | Different records or graph components do not prove independent experiments or causal replication |
| Local models | Manifest-verified local extraction and MiniLM retrieval; fresh execution and cache replay are distinguishable | Source guards can abstain; accepted output is not expert validation, and truncated/selected text gives bounded coverage |
| Review and calibration | Blinded packets, validated labels, agreement/adjudication and opt-in fitted calibration machinery | Real expert labels and scientifically justified calibration protocols remain external, pending inputs |

The modality layer has eight typed metadata families: bulk and single-cell transcriptomics, sequencing/genomics, clinical registry metadata, microbiome/metagenomics, proteomics, metabolomics and declared perturbational screens. The lead's `d84e439` repair adds explicit pooled CRISPR/RNAi metadata and preserves distinct precise assays and unknown biological units; it does not claim raw screen acquisition. GWAS summary statistics are not adopted without a source-specific allele, ancestry, cohort-overlap and endpoint contract. Precise phenotype/pathway matching is deferred where source mappings, rights or lineage remain unqualified; source context and unknowns survive. Generic field representation alone does not establish comprehensive ontology coverage.

Read `SOURCE_ASSISTED_PENDING_EXPERT_REVIEW`, `UNCALIBRATED_HEURISTIC`, `UNKNOWN`, `REQUIRES_INSPECTION` and partial/insufficient-coverage states literally. A high score is a review priority. It does not mean that a dataset can answer the question, has adequate power, represents a new discovery or supports a causal conclusion. Genuine human labels are optional for software completion under the governing specification, but required before stronger expert-validation claims.

## Offline installation from the final artifacts

Use these commands only after the final package and clean-install receipts identify the delivered wheel and dependency bundle. The paths below are placeholders to replace with those actual artifact locations. The final wheel, sdist/source archive, their manifests and hashes must agree with the acceptance receipt. Creating this document does not establish that those artifacts have passed.

The package metadata supports Python >= 3.10; current validated development/runtime checkpoints use Windows Python 3.12.13. A base install needs the declared core dependencies. Model execution additionally needs an explicitly selected compatible PyTorch wheel and Transformers; model weights are separate local artifacts. An offline install requires every required dependency wheel to be present locally.

```powershell
$ErrorActionPreference = 'Stop'
$ldmWheel = 'C:\path\to\litdatamatcher-0.3.0-py3-none-any.whl'
$ldmWheelhouse = 'C:\path\to\verified-dependency-wheels'
$ldmEnvironment = 'C:\LitDataMatcher\venv-0.3.0'
py -3.12 -m venv $ldmEnvironment
if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed.' }
$ldmPython = Join-Path $ldmEnvironment 'Scripts\python.exe'
& $ldmPython -m pip install --no-index --find-links $ldmWheelhouse $ldmWheel
if ($LASTEXITCODE -ne 0) { throw 'Offline wheel installation failed.' }
& $ldmPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Installed dependency consistency failed.' }
& $ldmPython -c "from importlib.metadata import version; print(version('litdatamatcher'))"
if ($LASTEXITCODE -ne 0) { throw 'Installed package import failed.' }
```

The existing qualified model pair is Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28` and all-MiniLM-L6-v2 revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Each directory needs its actual `MODEL_MANIFEST.json` and all hash-matching model/tokenizer/config files. Runtime calls use local-only loading; a missing model is not silently downloaded. Model/runtime dependencies must be supplied in a compatible local wheelhouse or an independently validated runtime environment. See [runtime qualification](RUNTIME_QUALIFICATION.md) for the historical qualified Windows CUDA configuration and its explicit limits; its older continuation/policy notes are historical.

## Run a bounded local analysis

Use a prepared, writable data root with normalized literature in `catalog/literature.jsonl`, study metadata in `catalog/studies.jsonl`, and the corresponding qualified reference/HTTP caches. A new empty directory has no acquired evidence. `sync --offline` can replay existing source caches; it cannot manufacture uncached sources. Connected acquisition is a separate explicit operation using the supported source commands and recorded limits.

The installed `litdatamatcher-v2` entry point exposes `doctor`, `sync`, `analyze`, `reference-sync`, `report`, `acceptance` and `closeout-audit`. For a local document, explicit question and scientist-owned requirement file:

```powershell
$ldmCli = Join-Path $ldmEnvironment 'Scripts\litdatamatcher-v2.exe'
$ldmData = 'C:\LitDataMatcher\prepared-data'
$ldmExtractor = 'C:\path\to\Qwen2.5-7B-Instruct\a09a35458c702b33eeacc393d103063234e8bc28'
$ldmEmbeddings = 'C:\path\to\all-MiniLM-L6-v2\1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
$ldmRun = 'C:\LitDataMatcher\runs\my-new-run'
& $ldmCli doctor --root $ldmData
if ($LASTEXITCODE -ne 0) { throw 'Runtime inspection failed.' }
& $ldmCli analyze --root $ldmData --out $ldmRun --model $ldmExtractor --embeddings $ldmEmbeddings --document 'C:\path\to\paper.txt' --question 'Your explicit scientific question' --requirements 'C:\path\to\requirements.json' --limit 1 --chunks 2 --device cpu
if ($LASTEXITCODE -ne 0) { throw 'Analysis failed or returned a non-success status; inspect the run diagnostics.' }
& $ldmCli report --run $ldmRun
if ($LASTEXITCODE -ne 0) { throw 'Report generation failed.' }
```

Use a new output directory for each new run. The `--fresh` flag explicitly requests fresh model execution rather than eligible inference-cache reuse; `--device cuda` requires a qualified compatible GPU runtime. A requirement JSON is an array of records such as `{"field":"species","expected":"Homo sapiens","essential":true,"source_locator":"the user's explicit question"}`. Essential requirements must come from the user or a justified scientific contract; model field proposals require review before becoming hard filters.

Inspect `RUN_MANIFEST.json`, source/inference coverage, retained failures, `questions.jsonl`, `matches.jsonl`, `evidence_bundles.jsonl` and the rendered report before interpreting results. Dossiers retain missing requirements, provenance, dependence and ranking rationale. Empty or rejected model output must remain an abstention or partial result. Neither the historical alpha acceptance nor the CPU metadata benchmark certifies the pending six-case final execution.

## Final acceptance and roadmap

The release remains pending until actual final receipts establish the current full suite, changed-path/adversarial regression, completed real-case disposition, comparable fixture performance, package integrity, outside-checkout clean installation, installed offline smoke, machine acceptance, independent review and lead closeout/remote state. Preserved component passes support those gates but cannot replace them. The matrix keeps operational and scientific-validation statuses separate.

Further work should follow demonstrated need:

1. Resolve the current source/runtime/case findings and finish the final artifact and review gates. Recent source/schema and model-abstention findings mean that a campaign-wide no-further-gain claim is not yet justified by this draft.
2. Obtain genuine blinded expert judgments when available. Define the target outcome, study/cohort grouping and validation protocol before fitting calibration. The new machinery makes this possible; it supplies neither human judgment nor a scientifically justified probability model.
3. Qualify additional repositories or ontologies when they enable a specific question class with inspectable measurements, controls, units and lineage. Existing rights/lineage/access decisions for cadmus, interaction_finder, OptimusKG, PrimeKG, SNACKKSS and Reactome remain evidence-based component dispositions. Source-count growth alone is not the objective.
4. Extend source-qualified identifiers and modality vocabularies only from actual source contracts. Preserve ambiguous/unmapped fields until evidence supports a mapping. Broad graph imports or apparent matches should not replace measured sample-level compatibility.
5. Expand held-out answerability and calibration research under a new scientifically approved protocol, keeping the completed alpha holdout sealed. The 17 metadata queries do not support population accuracy, scientific-significance probabilities or global novelty estimates.

The completed benchmark gives no engineering reason for an immediate index/database redesign at the observed size: query p95, matching throughput, memory and recovery satisfy their predeclared bounds. Compatibility-only, compatibility-plus-lexical and compatibility-plus-MiniLM tie on the current source-metadata questions, so that result supplies no measured gain from adding another ranking layer. These are concrete local diminishing-return observations, not a claim that all future scientific work is exhausted. Human validation, repository measurement access, dependence annotations and genuinely new question classes are separate future evidence needs. The final stopping decision still requires the remaining current gates to pass or receive a governing, evidence-based disposition.
