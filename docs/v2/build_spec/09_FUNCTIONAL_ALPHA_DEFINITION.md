# Two delivery milestones, one complete assignment

## Milestone F: first functional alpha

Make a genuine end-to-end result available early. Minimum real coverage:

- 50 unique real literature records, including 20 successfully parsed full texts.
- 50 unique accession-level studies, with 20 sample/group-level capability profiles.
- One real processed dataset file inspected and linked to its sample annotations.
- One external structured evidence resource actually imported and queried.

Demonstrate automatic literature-driven questions and explicit-question matching. Produce traceable positive, negative, unknown and partial-fit examples, at least one complete question-to-data-to-evidence dossier, and an HTML report plus machine-readable records. Fresh semantic inference must occur in the application. Sources are not synthetic examples or the legacy curated repository summaries.

Label this `FUNCTIONAL_ALPHA_AVAILABLE`, publish a compact interim artifact, and continue. The full hardening/refinement assignment is not complete at this milestone. The alpha may still expose failures outside its declared supported path, but it must not disguise those as successful results.

## Milestone H: hardened alpha and closeout

All ten workstreams run in the integrated product. Minimum expanded coverage:

- 200 unique real literature records, including 50 parsed full texts.
- 100 unique accession-level studies, with 30 sample/group-level capability profiles.
- Processed files from at least two distinct real studies inspected with sample/feature validation.
- At least one versioned external structured evidence resource imported, queried, and lineage-audited.
- Literature acquisition and dataset acquisition can run separately and update the shared catalog; at least GEO and SRA or ENA data paths are exercised.
- Two distinct pilot topics/contexts, one reserved for transfer evaluation. They may share biological scope but cannot merely paraphrase the same questions.
- At least six end-to-end case dossiers collectively covering direct fit, partial/unknown fit, no-fit, already-answered/changed gap, contradictory or indirect evidence, and a multi-source lineage/integration case. A single dossier may cover several dimensions, but there must be six distinct question/dataset assessments.

The default primary pilot is human inflammatory-response perturbational transcriptomics; a related gut/IBD transcriptomics context is a reasonable transfer pilot. These are engineering proving grounds, not permanent application eligibility rules or claimed research findings. Codex may choose equivalent accessible subjects within this scope based on actual source availability, document the choice before evaluation, and avoid cherry-picking only known easy successes.

Count unique studies/cohorts at the scientific unit level, not API rows, mirrored repositories, SRA runs, article versions or repeated files. Keep denominators and exclusions. Source replacement may solve availability problems; do not invent coverage or retroactively reduce floors. Collection floors alone do not establish scientific quality; file 08's executed gates also apply.

## What the user receives

An installable/runnable local application with tested commands for environment diagnosis, acquisition/update, document/topic/question input, analysis, evaluation, report generation, validation, resume and stop. Equivalent existing command names are fine; do not claim example commands exist before implementing and testing them.

A real-data local HTML report/review interface provides ranked questions, candidate data, experimental fit, evidence dependency groups, contradictory/indirect evidence, source locators, missing requirements, component scores, and the proposed next analysis. A new user input must work without editing Python source.

Delivery contains source/package version and commit, environment lock, code/model/source/data manifests, notices, configuration without secrets, small permissible fixtures, benchmark results, worker scorecards, independent reviews and `ACCEPTANCE_REPORT.json`. Large data/models remain outside the code ZIP, with exact local references or permitted retrieval/rebuild instructions.

## Readiness statuses

Product: `NOT_READY`, `FUNCTIONAL_ALPHA_AVAILABLE`, `HARDENED_ALPHA_READY`, or `BLOCKED_EXTERNAL_PREREQUISITE`.
Development automation: `NOT_CONFIGURED`, `PREPARED_NOT_ENABLED`, `VERIFIED_ENABLED`, `PAUSED`, `UNAVAILABLE`, or `DISABLED_AT_COMPLETION`.
Scientific calibration: `PENDING_EXPERT_LABELS`, `SOURCE_ASSISTED_EVALUATION`, or `EXPERT_CALIBRATED` with actual supporting evidence.

Report these independently. Pending expert labels do not prohibit a functional engineering alpha with clearly labeled heuristic rankings. A fake model backend, absent real data, failed required gate or invented human validation does prohibit the corresponding readiness claim.
