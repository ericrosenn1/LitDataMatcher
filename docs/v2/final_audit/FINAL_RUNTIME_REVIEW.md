# Independent runtime follow-up review

Status: **PASS_WITH_LIMITATIONS** for the bounded runtime scope. Four reproduced findings are fixed and independently validated: 0 critical, 1 high, 3 medium, 0 unresolved. This does not certify the final assembled package or scientific case contracts.

Reviewed source: `8001e7e64e8cbf7ba24b2a1b2119a745a28dc1c0`; authoritative root repair: `108cf4c85c9be663a2ed0d5bbba83abc7761d638`. The five reviewed core source blobs match that root repair exactly. Source fingerprint: `459fca8416c81f1482365647fbebdc39fa2be80c538c0d4a7a1a58899e330569`. Baseline: `bbe5c28868bfc7e564ec0906174642c7f22f7dbf`. Reviewer `/root/final_offline_repair`; implementer `/root`; expert status `PENDING_EXPERT_REVIEW`.

| Finding | Severity | Reproduced defect | Repair and validation |
|---|---|---|---|
| RFR01 | Medium | Explicit review/meta-analysis findings accepted as current experiments | d2487ec rejects explicit synthesis roles; two red probes now pass. The real metabolic case is now background. |
| RFR02 | Medium | Extra model fields could assert novelty, expert validation or calibration | d2487ec rejects undeclared claim fields; two red probes now pass. |
| RFR03 | Medium | Retrieved context dropped a known publication date | 15902b5 retains the supplied date; future-dated context is excluded by the compiler. |
| RFR04 | High | Six exported dossiers could mask failed analysis execution | 15902b5 and 108cf4c propagate failures; only matching PASS/no-failure or PARTIAL/source-guard-only statuses qualify. FAIL, inference, runtime, artifact and mixed-stage probes fail closed. |

**Executed validation:** 29 independent runtime fixtures and 100 related fixtures passed on Python 3.12.13, with zero network attempts. Runtime fixtures deny model loading. Red receipts preserve the original 18-test result (4 failures), context/controller 24-test result (2 failures), and intermediate 29-test controller result (4 failures). Exact commands, log/JUnit hashes, test-file hashes and source commits are in the JSON receipt.

The transport controls accept only complete raw JSON or one complete JSON fence and reject prose, truncation, wrong-language fences, trailing commas and extra objects. Source-binding controls reject retracted/unselected IDs and retain scoped context without direct-support or novelty claims. Observed abstention/rejection may retain exact unasserted source spans; a thrown inference failure produces no phantom dossier.

**Observed application results:** preserved fresh_run2 failed with two accepted claims and two exported cases. Preserved fresh_run3 exported six cases, with two accepted claims and four source-context-only cases. All six native statuses are PARTIAL. All nine recorded failure entries are source_guard rejections; no runtime/inference/artifact failure was observed. The 90 individual dossier rows are repeated question/candidate combinations, not 90 independent scientific cases.

All inspected input/artifact hashes, accepted claim spans and deterministic context spans match. Retrieved context is background/inconclusive, NOT_ASSERTED, has no claim object, and cannot answer the question. All evidence bundles remain CONTEXT_ONLY_OR_UNRESOLVED with insufficient-coverage. The two accepted claim records also pass read-only revalidation with identical identities, parent offsets and semantic fields. Runtime source is identical after line-ending normalization: the actual execution writer has mixed endings (SHA256 c1fdeb3c10c705a3b74e6e552a274133eca014cd74e74adba487ada86ac0dd23), while the reviewer has LF-only bytes (4d5243c5b28622a3efde48ebf115117ec3644c7401a8435d6c5622dfe092cc1e). Exact executed source and both hashes are retained; cache hashes were not changed.

**Scientific limits:** the species-only requirements in fresh_run3 can produce an exact fit for an unrelated dataset, so that preserved run is not the final scientific example contract. Root is preparing separate source-backed requirements and a cache-replay derivative. This report does not certify that future derivative. The richer original G12 dossier classes remain separate frozen evidence. Six source-context exports do not establish extraction precision, recall, global novelty, independent support or expert validation.

No model inference, acquisition, frozen evaluation, holdout execution or input mutation occurred during this review. Failed attempts retain their actual source commits and remain preserved. Final package/source-snapshot review is still required before a final_independent_review_v1 wrapper can bind this bounded report; the JSON defines the exact required wrapper fields.

Evidence directory: `C:\Codex\LitDataMatcher-v2\data\final_campaign_20260913\runtime_followup`. Machine receipt: `FINAL_RUNTIME_REVIEW.json`. Source fingerprint, read-only real-case inspections, red/green logs and JUnit files are retained there. The compact handoff includes a manifest and archive reopen/hash verification.
