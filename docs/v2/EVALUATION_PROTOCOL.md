# Evaluation protocol EP-20260907-1

Frozen before v2 model/ranking tuning by the independent evaluation worker at source checkpoint `0e1943d415949c6e2e302e3c2594a7049ea1ab20`. This is a prospective protocol, not validation evidence. The finalized build specification remains authoritative for coverage and readiness.

## Source selection and separation

The primary context is human inflammatory-response perturbational transcriptomics. Development anchors are GEO GSE193336 (human macrophage LPS), GSE128885 (deferiprone with/without LPS), GSE99787 (mouse macrophage LPS hard negative), GSE133844 (genotype comparison under LPS), and GSE95435 (human colon single-cell donor/unit challenge). These were chosen from source discovery, before inspecting generated questions or ranking results. Each fact used as a label must be checked against an immutable source snapshot and its exact field/section. Generated question attractiveness cannot determine inclusion.

Reserve the complete GSE112372 publication/cohort family for final primary holdout. Reserve GSE214695 and GSE226875 families for the distinct gut/IBD transfer evaluation. Only discovery summaries have been seen at freeze: no generated outputs or detailed labels were inspected. Reservation is not yet proof of source disjointness. Before any final claim, resolve DOI/PMID/PMCID versions, superseries/subseries, BioProject/SRA/ENA mirrors, reused donors/cohorts, and source-of-source links into connected components. Exclude every linked family from tuning. Unknown overlap remains unknown and disqualifies a claim of proven independence.

Additional evaluation sources are selected from acquisition manifests by stable accession/identifier order, within scope, before model output review. Freeze an expanded family manifest before evaluation: at least 20 fully judged primary queries and 10 transfer queries. Include candidates without direct accession citations and report them separately from linked rediscovery. No query may disappear after an unfavorable result. Sources unavailable through permitted routes retain an unavailable record; replacements and selection reason are logged before outcomes are seen.

Final holdout labels/results remain sealed outside Git until both development rounds with no material gain are complete. Reading holdout predictions for repair contaminates the family: relabel it development and select a new untouched family prospectively. The transfer pilot cannot be used for ordinary tuning and still be reported as fresh transfer. This build does not claim a historical temporal novelty evaluation from current databases.

## Label provenance and denominators

Allowed origins are `expert`, `source_determined`, `model_assisted`, and `unreviewed`. This protocol starts with source-determined design constraints, labeled by a Codex evaluator from cited repository fields; these are not expert gold. Explicit counterfactual modifications of real records are marked `source_derived_metamorphic` in construction metadata and never counted as acquired real records. Biological relation interpretations requiring judgment are model-assisted unless checked by an actual qualified human. Expert ranking calibration is pending.

For each case retain case ID, source family, snapshot SHA256, locator, question/requirements, judged candidate universe including negatives, expected safe behavior, origin/method, uncertainty, and split. Unknown is not negative. Missing labels are not zeros. Extraction recall requires independently enumerated source propositions; otherwise report precision/support only and recall NOT_RUN. Report every metric numerator/denominator and family count, with uncertainty intervals where meaningful. A small case suite cannot establish population accuracy.

## Predeclared quality gates

These additional pilot quality targets are risk-based engineering criteria, not biological power thresholds. Baseline results must be recorded before candidate comparison; the thresholds may not be silently revised after failure.

* Zero forbidden direct-fit promotions across the challenge set and judged top results: indispensable species/tissue/assay/comparator mismatch, unknown promoted to verified, donor count inferred from cells/runs, unpaired union promoted to joint observation, or incompatible pooling. Any such error is high severity and blocks the declared path.
* Zero source-span/identity corruption, duplicate-lineage increase in independent support, unsupported novelty from absence of hits, or reversed consequential negation in scored challenge claims.
* At least 95% source-supported qualifier and capability-field correctness on at least 40 independently enumerated fields spanning at least 10 source families. Counts under these floors are exploratory, not a PASS. Report each field family separately so easy identifiers do not hide comparator/negation failures.
* Candidate recall@10 at least 90% over at least 20 fully judged primary queries with at least one usable candidate; include all no-fit queries separately. Compatibility-aware invalid-top-match rate must be zero over the complete judged universe. Report nDCG@5 and precision@5 as descriptive metrics with graded source-determined fit labels (direct=3, partial=2, requires-inspection=1, not-qualified=0); these relevance labels are not truth probabilities.
* Run lexical-only, pretrained semantic/hybrid, and compatibility-aware methods on identical queries and candidate universes. Record runtime model/config and cache origin. The compatibility-aware method must reduce or tie invalid-top rate without losing a known direct candidate from retrieval; it need not fabricate an nDCG improvement when the baseline already ties.
* Trace every dossier claim and capability to persisted source locators; retain contradictory/indirect evidence and the compiler's before/after dated gap assessment. At least six distinct final dossiers must cover the specification's outcome types.

## Refinement and stopping

A material gain is any reproducibly fixed high-severity defect, at least 2 percentage points in a primary known-denominator quality metric, or at least 10% lower median latency on the same workload with no quality regression. For small samples, one corrected consequential error remains material; report absolute counts. Use the same frozen development set to compare incumbents and retain rejected candidates. A speed gain never offsets worse forbidden-promotion or source-support results.

Each central worker needs two substantive evaluations with new discriminating evidence; the integrated product needs three distinct rounds. Repeating identical passing tests alone is not a pass. Only after all required product gates pass and two consecutive development rounds produce no material gain may untouched holdout and independent closeout proceed. Any major holdout defect reopens targeted repair and holdout reservation.

## Execution and limits

Unit/metamorphic challenges verify contracts but do not replace live/replay, fresh semantic inference, independent review, clean installed distribution, blocked-network replay plus fresh inference, two-stage interruption/resume, resource-pressure tests, or machine acceptance. Preserve hashes and command exit codes. Keep raw source captures at the shared evaluation data root outside Git. Reports distinguish engineering readiness, supervision readiness, and calibration. Until source linkage, expanded judgments, and those executed gates exist, readiness remains incomplete.
