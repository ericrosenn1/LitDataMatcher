# LitDataMatcher 0.3.0: release, use and qualified scope

Version 0.3.0 adds qualified multisource metadata, explicit experimental requirements, source-linked evidence dossiers, strict review/calibration contracts and measured local retrieval and recovery. The frozen functional revision is `7bc86f8b81f33d00b8d40d7378e01d551b0266fa`, with normalized functional fingerprint `d3c546f9ffefaff42535b32314b32add73470be7e939682b158c877de166b028`. Documentation and distribution metadata can advance without changing those tested functional bytes. The [74-row requirements matrix](final_audit/FINAL_REQUIREMENTS_MATRIX.md) records final review and operational closeout separately; this document makes no public-release or expert-validation claim.

## What changed

The application now connects qualified repository metadata to explicit experimental requirements, preserves unknown metadata and biological independence, and produces source-linked evidence bundles and review dossiers. Recent repairs address identifier/lifecycle reconciliation, source-specific correction/retraction notices, per-response-page provenance, offline cache isolation, atomic cache writes, strict adjudication and unsupported calibration claims. Controlled organism aliases were added only after source qualification and a reproduced real benchmark failure.

The expanded acquisition contains 1,600 unique literature identifiers and 487 catalog accessions across cancer, environmental microbiome, metabolic and neurologic contexts. A separate qualification adds 65 proteomics/metabolomics accessions. The combined 552 catalog identifiers comprise 395 ClinicalTrials.gov, 75 MGnify, 17 ENA/SRA, 25 PRIDE and 40 Metabolomics Workbench records. These are source identifiers, not 552 proven independent cohorts. The literature sample contains 1,456 abstracts and 144 metadata-only records. It is not a new 1,600-full-text corpus.

All 13 expanded acquisition partitions replayed offline. Three omics partitions replayed identically on final source with zero unexpected network requests. The final real CPU benchmark passed at 100/32, 500/100 and 1,600/552 literature/catalog sizes. At full size, warmed FTS p95 was 18.11 ms, matching took 0.0444 seconds per 1,000 actual candidate assessments, and sampled peak process-tree RSS was 1.166 GiB. A child committed 276 catalog records and exited 71; a different child resumed only the remaining 276, with exact IDs/payloads and no duplicate versions. All nine native benchmark source hashes match frozen Git bytes. These are engineering results for the declared local metadata workload. The initial failure and earlier checkpoints remain preserved. [Final benchmark receipt](C:/Codex/LitDataMatcher-v2/data/final_campaign_20260913/closeout/scale_real_delivery/BENCHMARK_RECEIPT.json), [final omics replay](C:/Codex/LitDataMatcher-v2/data/final_campaign_20260913/closeout/OMICS_REPLAY_FINAL.json).

The benchmark's 17 source-derived metadata questions evaluate 9,384 pairs: 1,376 observed fits, 6,439 confirmed metadata mismatches and 1,569 unknown pairs. Wrong-modality and wrong-organism counts overlap. Unknowns remain unjudged rather than being silently counted as negatives. Compatibility-aware variants tied and produced zero confirmed-invalid top results; lexical and MiniLM retrieval produced two and three respectively. Queries and reference labels derive from the same catalog with a bounded vocabulary, so this is not held-out biological answerability or calibrated scientific accuracy. [Benchmark protocol and earlier repaired-run report](final_audit/PHASE2_BENCHMARK_RESULT.md).

Eight real UniProt entries now have qualified records, exact source-derived alias queries and network-denied replay: INS, APP, MAPT, MTOR, PDCD1, TREX1, TREM2 and CD274. The receipt retains the actual accessions, release, source/license snapshots and hashes. These queries provide reference context; they do not establish pathways, experimental design or independent biological support.

Six new source-selected cases across four domains now use **23 requirements supported by exact retained passages**. Corrected replay preserves all six source identities, questions, model outputs and the 552-candidate catalog. Two model claims passed source guards; four cases retain source context only. All six native runs remain `PARTIAL`, with source-guard failures visible. The final execution passes its declared case contract while retaining `UNKNOWN`/`REQUIRES_INSPECTION` outcomes. The observational-versus-randomized-trial negative is corrected, all six source dates are now qualified, and duplicate merging reduces the metabolic case from three entries to two unique questions. No positive answerability or expanded extraction-accuracy claim is inferred. [Final case manifest](C:/Codex/LitDataMatcher-v2/data/final_campaign_20260913/closeout/CASES_MANIFEST_DELIVERY.json), [native replay receipt](C:/Codex/LitDataMatcher-v2/data/final_campaign_20260913/closeout/case_replay_delivery/CASE_EXECUTION.json).

The original hardened-alpha six-class dossier coverage and acceptance remain frozen. Original alpha G12 concerns failure recovery. The later six-case plan is a separate execution floor for Phase2 scientific breadth; it does not replace alpha evidence or establish an extraction precision/recall denominator. Context is never promoted to an accepted structured claim. Earlier independent functional review closed six findings with 45 adversarial and 174 related checks on its recorded source; final independent artifact/case review remains a distinct gate.

The final functional source passed **699 unique full-suite tests and 196 overlapping focused runtime/review/validator checks**, with zero failures, errors or skips. The unchanged 32-record synthetic performance fixture also passed its original baseline and tolerance, separately from real scale. Targeted repairs cover lifecycle signals, response-page provenance, cache isolation/atomic writes, adjudication, exact organism mappings, duplicate questions, qualified randomized-trial requirements and acquired publication-date fallback. The last repair promotes only valid full ISO `metadata.first_publication_date` from exact EuropePMC sources, retaining provenance and explicit top-level dates; missing, partial or unqualified dates remain unknown. [Full-suite receipt](C:/Codex/LitDataMatcher-v2/data/final_campaign_20260913/closeout/full_suite_delivery.command.json), [focused receipt](C:/Codex/LitDataMatcher-v2/data/final_campaign_20260913/closeout/focused_suite_delivery.command.json), [fixture receipt](C:/Codex/LitDataMatcher-v2/data/final_campaign_20260913/closeout/SCALE_FIXTURE_FINAL.json).

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

The versioned wheel, sdist and source ZIP are recorded in [build validation](C:/Codex/LitDataMatcher-v2/data/final_campaign_20260913/release_0.3.0_delivery/BUILD_VALIDATION.json); their hashes, members and ZIP CRC were checked. The actual wheel installed outside the checkout and passed both CLI entry points, demo, v2 doctor and report. Installed offline `analyze` replay reproduced a nonempty scientific payload from an actual earlier fresh local run exactly, with two cache hits, zero fresh model calls, one blocked network control probe and zero unexpected network attempts. This is installed replay evidence, not a fresh installed model call. [Install receipt](C:/Codex/LitDataMatcher-v2/data/final_campaign_20260913/closeout/install_delivery.command.json), [offline receipt](C:/Codex/LitDataMatcher-v2/data/final_campaign_20260913/closeout/offline_delivery.command.json). Select the delivered wheel and dependency bundle by the final acceptance-receipt hashes; replace the example paths below with those locations.

The package declares Python >= 3.10; actual installation/runtime validation used Windows Python 3.12.13. Python 3.10 syntax was checked separately, without claiming an actual 3.10 installation. The installed runtime used PyTorch 2.11.0+cu128 and Transformers 4.57.6, supplied from existing caches without new downloads. Core dependencies and model runtime wheels must be available locally for an offline install. [Installed runtime](C:/Codex/LitDataMatcher-v2/data/final_campaign_20260913/closeout/INSTALLED_RUNTIME.json).

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

Inspect `RUN_MANIFEST.json`, source/inference coverage, retained failures, `questions.jsonl`, `matches.jsonl`, `evidence_bundles.jsonl` and the rendered report. Dossiers retain missing requirements, provenance, dependence and ranking rationale. Rejected model output remains an abstention or partial result. Required variables, outcomes and experimental conditions need an explicit source-supported contract; a species-only `EXACT_FIT` result does not adequately assess a biomarker or treatment-response question.

## Final acceptance and roadmap

Technical source regression, corrected case replay, package integrity, outside-checkout installation and installed offline smoke have actual receipts. Final independent approval, machine acceptance and lead state/remote/worker closeout remain separately tracked by the matrix. A diagnostic report with passing component gates is not itself an overall acceptance PASS. Final release documentation accompanies the frozen functional package and binds its exact artifacts; no successful historical scientific run is replaced.

Further work should follow demonstrated need:

1. Preserve the validated repairs and their failed predecessors. Exact source-bound requirements removed underconstrained case fits; a narrow qualified guard now rejects explicit observational/nonrandomized metadata against randomized-trial requirements. Missing allocation or measurements remain unknown. Duplicate question merging preserves lineage while removing redundant assessments.
2. Obtain genuine blinded expert judgments when available. Define target outcomes, cohort grouping and validation splits before fitting calibration. The implemented review machinery supplies neither human judgments nor a scientifically justified probability model.
3. Qualify repository measurements, controls, units, sample identities and cohort dependence for a specific question class. Adding accessions alone cannot resolve unknown candidate measurements or statistical adequacy.
4. Extend exact identifiers and assay/phenotype vocabularies from inspected source contracts. Preserve ambiguous fields, source rights and lineage. Existing component dispositions for cadmus, interaction_finder, OptimusKG, PrimeKG, SNACKKSS and Reactome remain evidence based; generic graph imports do not supply missing experimental evidence.
5. Evaluate extraction and biological answerability under a new scientifically approved protocol with a real denominator. Keep the completed alpha holdout sealed. The 17 metadata queries and four context-only cases do not establish population accuracy, scientific-significance probabilities or global novelty.

The completed delivery benchmark gives no engineering reason for an immediate index/database redesign at the observed size: query p95, matching throughput, memory and recovery satisfy their predeclared bounds. Compatibility-only, compatibility-plus-lexical and compatibility-plus-MiniLM tie on the current source-metadata questions, so that result supplies no measured gain from adding another ranking layer. These are concrete local diminishing-return observations, not a claim that all future scientific work is exhausted. Human validation, repository measurement access, dependence annotations and genuinely new question classes are separate future evidence needs. Final independent review and operational delivery remain separate gates; the matrix records their actual evidence.
