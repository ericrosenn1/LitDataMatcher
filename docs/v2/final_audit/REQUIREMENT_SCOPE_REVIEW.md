# Governing requirement scope review

Audit source: `98e6e95ad924caafdd1b7ec5c373e27aef57b923`, isolated branch `codex/final-spec-audit-20260913`. This is a requirements/evidence snapshot for lead integration, not final project acceptance. The companion `REQUIREMENT_SCOPE_REVIEW.json` contains 74 requirements, source lines and excerpts, evidence paths and hashes, classification, blocking flags, and remaining actions.

## Decision: human expert labels do not block software completion

The governing specifications explicitly permit software acceptance with expert calibration pending. Real independently supplied human labels are required before an **EXPERT_VALIDATED / EXPERT_CALIBRATED** claim; they are not a mandatory gate for the machine-validated software project. Keep `PENDING_EXPERT_REVIEW` visible, or `PENDING_EXPERT_LABELS` on the original calibration axis. This decision does not waive any other scientific, engineering, packaging, review, or Phase 2 requirement.

The controlling source trail is:

| Authority | Exact requirement and implication |
|---|---|
| `docs/v2/build_spec/01_MASTER_CODEX_BUILD_PROMPT.md:51,95` | Implement machinery for capabilities needing absent labels. Expert ranking calibration may remain pending with transparent heuristic scores at required delivery. |
| `docs/v2/build_spec/08_VALIDATION_AND_ACCEPTANCE_GATES.md:22,60-68` | G10 requires evaluation, label provenance, grouped holdout, hard negatives, source-disjoint tests and explained scores. Label origins may be `expert`, `source_determined`, `model_assisted`, or `unreviewed`; true expert calibration may remain pending. |
| `docs/v2/build_spec/09_FUNCTIONAL_ALPHA_DEFINITION.md:40-46` | Product, automation and scientific calibration are separate readiness axes. No invented expert validation or failed required product gate may be disguised as readiness. |
| `docs/v2/build_spec/16_RUNTIME_MODEL_AND_DATA_POLICY.md:35-39` | Build expert-review import/export and calibration machinery. Missing experts permit transparent source/model-assisted evaluation and heuristic ranking, without probabilities of truth or experimental success. |
| Phase 2 user attachment, lines 230-271 | Requires expanded ranking evaluation/calibration procedures, ablations and expert-review machinery. When expert review is unavailable, preserve the machinery and clearly label the remaining validation requirement. It does not revoke the explicit pending-calibration software-delivery allowance. |
| Phase 2 attachment, lines 416-442 | `V2.4-CALIBRATED-MATCHER` is a **suggested** milestone name. It cannot establish actual calibration or silently override the explicit missing-expert rule. The underlying planned workstreams still require implementation or evidence-based disposition. |
| `project_state/FINAL_CAMPAIGN_REQUEST_20260913.md:203-234` | Explicitly asks for this distinction. If expert labels are not mandatory, close legitimate machine-validated software with expert review pending; otherwise complete all machine work first. |
| `project_state/OWNER_UNRESTRICTED_OVERRIDE_20260913.md:124-163` | Repeats that conditional rule and prohibits fabrication or scientific-criteria changes. The override changes computational authorization only. |

The Phase 2 attachment was read in full at `C:\Users\eric\.codex\attachments\292f19f6-c61c-4e35-b4a5-b81aab461356\pasted-text.txt`. Its SHA-256 and exact relevant excerpts are retained in the JSON. This is a user-scope source; old handoff dependency prose is implementation history rather than a new acceptance rule.

Existing source-determined labels may legitimately satisfy G10 for source-supported design/compatibility judgments under the frozen evaluation protocol. Preserve origin, annotator/method, source fields, uncertainty, candidate denominators and negatives. A Codex evaluator's source-determined labels are not expert gold; inferred biological interpretations still need their actual model-assisted or human-reviewed provenance (`docs/v2/EVALUATION_PROTOCOL.md:15-19`).

## Frozen alpha is complete within its historical scope

The actual protected receipt `C:\Codex\LitDataMatcher-v2\data\release\ACCEPTANCE_REPORT_FINAL3.json` has all G01-G16 `PASS`, `HARDENED_ALPHA_READY`, no open issues, **`PENDING_EXPERT_LABELS`**, and an empty calibration-evidence array. Its SHA-256 is `dc1bc32317f1430c3a100207fa9ff8e3f1b2e9310783e044eda096a6d04cb384`.

Recorded hardened-alpha coverage is 200 literature records, 50 parsed full texts, 111 independent accession-study groups, 31 sample-profiled studies, 3 inspected studies, 1 structured resource, 2 contexts and 20 dossiers. The fixed floors remain 200/50/100/30/2/1/2/6. The accepted refinement records two worker passes, three integrated rounds, two no-material-gain rounds, independent review and the passed final holdout.

This audit independently read and hashed all seven artifacts listed in `data\phase2\validation\alpha_baseline_pre_next_window.json`; every current hash matched. The frozen baseline commit is `5747cbea2ae65c8570280d0e53f77bfabc968712`. Historical acceptance/delivery evidence names its own earlier source fingerprint; the audit preserves those distinctions rather than replacing them with the current HEAD.

No holdout was executed, rescored, tuned, replaced or edited. No source was retrieved and no protected artifact was changed. The frozen acceptance result is preserved; it does not validate the later changed package automatically.

## Phase 2 remains broader than the alpha and bounded tranches

The Phase 2 correction makes hardened alpha a required milestone, not the final endpoint (attachment lines 1-32, 448-465). Its A-O workstreams must be implemented **or explicitly rejected with evidence**, with larger-scale/multisource tests, green regressions, no unresolved critical/high defects, diminishing meaningful returns, and a documented roadmap. The attached matrix keeps all those obligations visible; it does not classify the whole Phase 2 plan as optional.

| Requirement | Inspected evidence | Remaining scope issue |
|---|---|---|
| A-B: literature/repository expansion | Europe PMC/Crossref one-record metadata qualification; ClinicalTrials.gov registry record; bounded ENA study; MGnify study; exact PMID/PMCID reconciliation | Reconcile all required source investigations, adopted-source contracts and integrated breadth. Optional candidate names are not a download census. |
| C-G: modalities, semantic interpretation, entities, evidence, requirements | Existing alpha and later contract/normalization/lineage/formalization handoffs; V2.2 and V2.3 receipts explicitly use synthetic fixtures | Link full declared Phase 2 capabilities to actual executed validation or evidence-based scope dispositions. Integrity-only guards do not validate semantic breadth. |
| H-I: ranking and review machinery | V2.4 synthetic scorecard receipt and a one-item fixture review receipt with zero expert labels | Repair proven machinery/status issues; reconcile larger evaluation and ablation evidence. Preserve human review as optional additional validation. |
| J: larger workloads | V2.5 32-record synthetic fixture and same-fixture comparison; no inference | This proves bounded instrumentation, not substantially larger corpus/catalog or inference performance. No new numerical threshold is invented by this audit. |
| K: adversarial testing | Cross-source receipt and prior 346-test full suite | Final-source regression after repairs. |
| L: diverse real case studies | V2.6 dossier-template receipt plus original alpha dossiers | Template success is not a diverse real Phase 2 case study beyond the alpha pilots. Locate or complete valid real evidence or documented scope disposition. |
| M-O: justified refactoring, reproducibility, refinement | Existing replay/manifests/baselines/repair history | No refactor for its own sake. Reconcile final reproducibility, Phase 2 iteration and diminishing-return evidence. |

A `MACHINE_COMPLETABLE_REMAINING` classification can mean incomplete **evidence/disposition reconciliation**, not necessarily missing code. The lead should use existing successful artifacts wherever sufficient. This audit's snapshot is deliberately conservative where only bounded receipts were inspected.

Component-level non-adoption has real recorded evidence: `docs/v2/EXTERNAL_COMPONENT_REUSE_MATRIX.tsv:2-7` documents scoped cadmus, interaction_finder, OptimusKG, PrimeKG, SNACKKSS and SNACKKSS_NLP decisions; `docs/v2/EXTERNAL_EVIDENCE.md:3-18` documents optional Reactome deferral and the qualified limited UniProt panel. These decisions neither require new installations nor reject an entire Phase 2 workstream by implication.

## Confirmed machine defects and claim limits

These findings concern promised validation machinery and remain machine-resolvable software work. The probes used explicitly synthetic in-memory fixtures only; no expert-label artifact was created.

1. **Adjudication validation:** `litdatamatcher/expert_review.py:151-165` accepted `decision=NOT_IN_VOCABULARY` with an empty `policy_id` and returned `ADJUDICATED` / `ADJUDICATED_POLICY_LABELS_ONLY`. The intended categorical vocabulary and explicit policy cannot be bypassed by a nonempty arbitrary string.
2. **Scorecard calibration claim:** `litdatamatcher/calibration_readiness.py:38-55` reports `CALIBRATED` once provenance-bearing source-determined rows include two classes. It computes fixed 0.5 accuracy and per-ablation counts, with no fitted/validated calibration or component-value comparison. A synthetic two-row probe returned `CALIBRATED` with accuracy 0.0. Poor accuracy itself is not the defect; unsupported calibration status is.
3. **Acceptance calibration promotion:** `litdatamatcher/acceptance.py:611-624` returns `EXPERT_CALIBRATED` solely because a manifest has label origin `expert`. The in-memory probe supplied no actual calibration result. Label provenance alone cannot establish calibration.
4. **Receipt meaning:** `v2_4_expert_review\packet_receipt.json` has one fixture item and zero expert labels; `v2_4_calibration_readiness\receipt.json` explicitly uses synthetic source-determined rows. They prove infrastructure behavior within those bounds, not expert agreement, expert gold or calibrated scientific rankings.

The existing `calibration.py:11-54` threshold utility is explicitly exploratory tuning/QA. It does not repair these overclaims by mere presence. A safe final state can retain real calibration pending while still making the machine interfaces valid and accurately named.

## Mandatory final campaign gates

The final request requires a reconciled matrix; focused, subsystem and full regressions; diff validation; offline replay; promised determinism; protected-asset integrity; cross-source adversarial tests; appropriate recovery and existing performance-baseline comparison; versioned wheel/sdist/archive construction; clean installation and installed offline CLI smoke; source-bound acceptance and hashed manifests; final independent adversarial review and repair; truthful state; lead commit/push and verified remote SHA (`FINAL_CAMPAIGN_REQUEST_20260913.md:147-185,277-294,345-498,504-533,615-698`).

The prior PMID/PMCID receipt reports 67 targeted and 346 full tests, zero failures/errors/skips, and seven protected hashes at functional commit `40921e58ca8a33e9de35847a116c324ec7d71500`. These are retained as prior bounded evidence. Old alpha package/install results cannot certify the new final source.

Current owner policy supersedes the old model/effort ceilings, duty-cycle waits and alpha-only stop instruction. No paid API/credits, supervisor restart, new schedule, force-push, main merge, public release, scientific threshold changes or holdout reruns are authorized. The lead still needs actual worker/supervisor/ownership observation at closeout; persisted state alone is not process evidence.

## Classification and validation

The JSON contains 28 `COMPLETE_VALIDATED`, 11 `COMPLETE_NEEDS_FINAL_REGRESSION`, 26 `MACHINE_COMPLETABLE_REMAINING`, 2 `OPTIONAL_FUTURE_VALIDATION`, 3 `SUPERSEDED`, and 4 `NOT_APPLICABLE` rows. There are 37 currently blocking machine/evidence-reconciliation rows and **zero established `HUMAN_EXTERNAL_REQUIRED` software gates**. Alpha-complete rows apply to the frozen milestone only; they do not silently mark changed final code validated.

All ten requested numbered build documents were read in full, along with BUILD_SPEC, the Phase 2 user attachment, current final request/owner policy, and relevant code/handoffs. Source lines/excerpts, evidence existence and digests, JSON structure/vocabulary, unique requirement IDs, fixed alpha statuses and the seven protected hashes were checked. No full scientific suite or holdout was rerun for this documentation audit. The audit is `PASS_WITH_LIMITATIONS`: the scope decision is complete, while final-source execution and full Phase 2 evidence disposition belong to the lead campaign.

The lead should integrate these two audit files, repair the confirmed machinery defects, resolve each remaining mandatory requirement using exact evidence, and perform final package/install/adversarial validation before deciding COMPLETE. No additional real-expert dependency should be invented from stale NEXT_ACTION prose.
