# Initial final contract review

Audit source: `98e6e95ad924caafdd1b7ec5c373e27aef57b923`. Status: **PASS_WITH_FINDINGS** (audit executed; declared software functionality still has open defects).
UTC: 2026-09-13T21:23:03.257862+00:00. Python 3.12.13, existing offline environment.

8 reproduced findings: 7 high and 1 medium. The existing focused tests still pass: **14 passed, 0 failures/errors/skips**. No source code, shared data, frozen alpha or sealed holdout was changed.

## Governing acceptance and scope

The final request requires repair of demonstrated defects in declared functionality. Original acceptance permits true expert calibration to remain pending (`docs/v2/build_spec/08_VALIDATION_AND_ACCEPTANCE_GATES.md:64`; master prompt `01:95`). Real expert labels are not required for a machine-validated software release, but the offered import/adjudication/calibration machinery must remain valid. No AI-generated expert labels were produced.

This review inspected actual APIs, optional-adapter CLI dispatch, v2 analyze/report call sites, Phase 2 handoffs and current tests. The lead owns packaging, final regression, artifact integrity and completion classification.

## Findings

### CA-01 [P1] Offline file POST still attempts network access

Location: `litdatamatcher/http_cache.py:174-183`. Requirement: `docs/v2/build_spec/02_FINAL_ARCHITECTURE_SPEC.md:103`, `docs/v2/build_spec/08_VALIDATION_AND_ACCEPTANCE_GATES.md:23`.

On a cache miss, CachedHttpClient(offline=True).post_file_text reaches requests.post. Offline mode does not constrain this public method.

Contract scope: Declared Phase 2 functionality

Executed output:

```text
RuntimeError NETWORK_CALL_INTERCEPTED requests_post_calls= 1
```

Bounded repair: Apply the offline cache-only guard before entering the POST request loop. Add cache-hit/cache-miss tests that forbid requests.post.

### CA-02 [P1] Adjudication accepts unsupported or contradictory decisions as eligible labels

Location: `litdatamatcher/expert_review.py:151-165`. Requirement: `project_state/V2_4_EXPERT_REVIEW_HANDOFF.md:5`, `project_state/FINAL_CAMPAIGN_REQUEST_20260913.md:332`, `docs/v2/build_spec/08_VALIDATION_AND_ACCEPTANCE_GATES.md:64`.

A decision outside LABEL_VALUES, an empty policy ID, or two conflicting decisions for one item/dimension are accepted and emit ADJUDICATED plus ADJUDICATED_POLICY_LABELS_ONLY. This bypasses categorical validation at the final label boundary.

Contract scope: Importable expert-review helper; currently called by tests, not by a production CLI. Optional real expert labels do not excuse invalid machinery.

Executed output:

```text
unsupported {"accepted_decisions": [{"decision": "invented", "dimension": "relevance", "policy_id": "p1", "provenance": {"dataset_provenance": [], "question_source_spans": []}, "rationale": "source fixture", "review_item_id": "blind_review_6a2c3b4cfb0e44e2"}], "calibration_eligibility": "ADJUDICATED_POLICY_LABELS_ONLY", "invalid_decisions": [], "status": "ADJUDICATED", "unresolved": []}
duplicate_conflict {"accepted_decisions": [{"decision": "relevant", "dimension": "relevance", "policy_id": "p1", "provenance": {"dataset_provenance": [], "question_source_spans": []}, "rationale": "source fixture", "review_item_id": "blind_review_6a2c3b4cfb0e44e2"}, {"decision": "not_relevant", "dimension": "relevance", "policy_id": "p1", "provenance": {"dataset_provenance": [], "question_source_spans": []}, "rationale": "source fixture", "review_item_id": "blind_review_6a2c3b4cfb0e44e2"}], "calibration_eligibility": "ADJUDICATED_POLICY_LABELS_ONLY", "invalid_decisions": [], "status": "ADJUDICATED", "unresolved": []}
missing_policy {"accepted_decisions": [{"decision": "relevant", "dimension": "relevance", "policy_id": "", "provenance": {"dataset_provenance": [], "question_source_spans": []}, "rationale": "source fixture", "review_item_id": "blind_review_6a2c3b4cfb0e44e2"}], "calibration_eligibility": "ADJUDICATED_POLICY_LABELS_ONLY", "invalid_decisions": [], "status": "ADJUDICATED", "unresolved": []}
```

Bounded repair: Validate policy identity, supported dimension/value, packet membership and decision uniqueness; keep disputed or invalid targets ineligible. Preserve decision provenance without exposing blinded reviewers.

### CA-03 [P1] Calibration scorecard marks malformed labels and denominators CALIBRATED

Location: `litdatamatcher/calibration_readiness.py:15-44`. Requirement: `project_state/V2_4_CALIBRATION_READINESS_HANDOFF.md:3`, `project_state/NEXT_ACTION.md:50`, `project_state/FINAL_CAMPAIGN_REQUEST_20260913.md:332`, `docs/v2/build_spec/08_VALIDATION_AND_ACCEPTANCE_GATES.md:22`.

Each independent mutation of an otherwise valid two-row fixture passes: arbitrary string provenance, empty split family, duplicate conflicting record identity, FAILED record status, or forbidden novelty dimension with trailing whitespace. All emit metrics and CALIBRATED with no reason codes.

Contract scope: Standalone scorecard API and receipt helper. It is not wired into calibrate-ranking; that older exploratory threshold report is a separate contract.

Executed output:

```text
invalid_provenance {"calibration_status": "CALIBRATED", "metrics": {"accuracy_at_0_5": 1.0, "denominator": 2, "negative_labels": 1, "positive_labels": 1}, "reason_codes": []}
blank_split {"calibration_status": "CALIBRATED", "metrics": {"accuracy_at_0_5": 1.0, "denominator": 2, "negative_labels": 1, "positive_labels": 1}, "reason_codes": []}
duplicate_identity {"calibration_status": "CALIBRATED", "metrics": {"accuracy_at_0_5": 1.0, "denominator": 2, "negative_labels": 1, "positive_labels": 1}, "reason_codes": []}
forbidden_dimension_whitespace {"calibration_status": "CALIBRATED", "metrics": {"accuracy_at_0_5": 1.0, "denominator": 2, "negative_labels": 1, "positive_labels": 1}, "reason_codes": []}
failed_record_status {"calibration_status": "CALIBRATED", "metrics": {"accuracy_at_0_5": 1.0, "denominator": 2, "negative_labels": 1, "positive_labels": 1}, "reason_codes": []}
```

Bounded repair: Validate retained status, structured source provenance, nonblank split identity, consistent unique observation keys and normalized supported dimensions before calculating any denominator. Do not fabricate labels or change scientific thresholds.

### CA-04 [P1] Adapter gate rejects a matching RNA-seq assay and established organism alias

Location: `litdatamatcher/scientific_v2.py:182-193`. Requirement: `docs/v2/build_spec/08_VALIDATION_AND_ACCEPTANCE_GATES.md:20`, `project_state/V2_REQUIREMENT_FORMALIZATION_HANDOFF.md:3`, `project_state/V2_2_ELIGIBILITY_INTEGRATION_HANDOFF.md:3`.

An observed assay of RNA-seq with assay_types=[RNA-seq] becomes NOT_QUALIFIED for an RNA-seq requirement: the hard gate compares the raw assay to the bulk_transcriptomics family. A human/Homo sapiens exact alias also mismatches before established normalization.

Contract scope: Declared Phase 2 functionality

Executed output:

```text
assay {"compatibility_status": "INCOMPATIBLE", "eligibility": "NOT_QUALIFIED", "requirements": [{"essential": true, "expected": "RNA-seq", "field": "assay", "observation": {"mapping_type": "exact", "reason": null, "source_locator": "fixture:field", "status": "observed", "value": "RNA-seq"}, "source_locator": "user question", "status": "MISMATCH"}]}
organism {"compatibility_status": "INCOMPATIBLE", "eligibility": "NOT_QUALIFIED", "requirements": [{"essential": true, "expected": "human", "field": "organism", "observation": {"mapping_type": "exact", "reason": null, "source_locator": "fixture:field", "status": "observed", "value": "human"}, "source_locator": "user question", "status": "MISMATCH"}]}
```

Bounded repair: Compare like representations in the hard gate; retain accepted exact/synonym organism and assay mappings without permitting wrong-modality records to qualify.

### CA-05 [P1] Dossier rejects the compiler standard scoped novelty disclaimer

Location: `litdatamatcher/scientific_dossier.py:13-14`. Requirement: `project_state/V2_6_DOSSIER_HANDOFF.md:3`, `docs/v2/build_spec/02_FINAL_ARCHITECTURE_SPEC.md:5`, `docs/v2/build_spec/08_VALIDATION_AND_ACCEPTANCE_GATES.md:27`.

build_dossier rejects the ordinary output of compile_evidence because its standard no-global-novelty disclaimer contains the substring global novelty. The existing dossier fixture uses a different sentence, masking the integration failure. validate_dossier repeats the same substring rule at line 21.

Contract scope: The V2.6 template is importable standalone machinery. This finding requires API composition, not a new command or wholesale replacement of the existing v2 report renderer.

Executed output:

```text
compiler_novelty_claim= Limited to recorded searched coverage; no global novelty assertion
ValueError Dossier cannot assert global novelty
```

Bounded repair: Use a compatible scoped novelty contract and explicitly test compile_evidence -> build_dossier -> validate/render. Continue rejecting positive unsupported novelty claims.

### CA-06 [P2] Literature derivation key ignores changed source content and snapshot hash

Location: `litdatamatcher/literature_integrity.py:76-88`. Requirement: `project_state/V2_LITERATURE_INTEGRITY_HANDOFF.md:3`, `docs/v2/build_spec/02_FINAL_ARCHITECTURE_SPEC.md:35`, `docs/v2/build_spec/13_RECOVERY_AND_RESUME_POLICY.md:11`, `docs/v2/build_spec/08_VALIDATION_AND_ACCEPTANCE_GATES.md:23`.

Changing an abstract from a positive statement to its negation while changing the cached source-content digest returns UNCHANGED and no affected derivations. The key covers only source IDs, relation names/data and lifecycle state. Missing keys also compare equal; this was not separately promoted to a finding.

Contract scope: Bounded defect in the public literature helper. Search finds invalidate_affected_derivations used only in tests. Catalog.upsert separately hashes complete payloads and has content-invalidation tests; this finding does not prove Catalog invalidation fails or require wiring every helper into it.

Executed output:

```text
{"current_key": "4af79790b8254b28696f3d9cae2753191473753755e99155707e51a9529c1989", "derivation_ids": [], "previous_key": "4af79790b8254b28696f3d9cae2753191473753755e99155707e51a9529c1989", "status": "UNCHANGED"}
```

Bounded repair: Include the stable content/source-version inputs promised by this helper, excluding volatile retrieval-only metadata where appropriate. Retain order-invariant identity and unchanged-replay tests; preserve unaffected derivations.

### CA-07 [P1] Empty offline cache becomes successful zero-result CLI search

Location: `litdatamatcher/adapters.py:952-961`. Requirement: `litdatamatcher/cli.py:100`, `project_state/V2_1_PAGINATION_HANDOFF.md:5`, `project_state/FINAL_CAMPAIGN_REQUEST_20260913.md:322`, `docs/v2/build_spec/08_VALIDATION_AND_ACCEPTANCE_GATES.md:24`.

Europe PMC and ClinicalTrials pagination errors disappear when there are no rows to carry their metadata; Crossref exceptions are similarly swallowed by search_literature_sources. The public CLI then writes an empty result and returns 0 without any error/status. Cache failure is indistinguishable from a successful empty query.

Contract scope: Declared Phase 2 functionality

Executed output:

```text
literature-search europepmc exit= 0 written_rows= 0 stdout= {
  "out": "unused.jsonl",
  "rows": 0,
  "sources": [
    "europepmc"
  ]
}
literature-search crossref exit= 0 written_rows= 0 stdout= {
  "out": "unused.jsonl",
  "rows": 0,
  "sources": [
    "crossref"
  ]
}
dataset-search clinicaltrials exit= 0 written_rows= 0 stdout= {
  "out": "unused.jsonl",
  "records": 0,
  "sources": [
    "clinicaltrials"
  ]
}
```

Bounded repair: Preserve query-level failure/partial status independently of result rows, or raise a typed failure on unsuccessful empty acquisition. Ensure CLI exits/status distinguish cache failure from genuine empty success while retaining valid partial records.

### CA-08 [P1] Failed cache refresh can destroy the previous replayable entry

Location: `litdatamatcher/http_cache.py:86-86`. Requirement: `project_state/V2_1_CACHE_REFRESH_HANDOFF.md:5`, `docs/v2/build_spec/13_RECOVERY_AND_RESUME_POLICY.md:13`, `docs/v2/build_spec/08_VALIDATION_AND_ACCEPTANCE_GATES.md:24`.

Refresh replaces the cache with Path.write_text in place. A simulated interrupted disk write truncates the old JSON; the next offline replay raises JSONDecodeError. Failed refresh therefore does not preserve the old entry as promised.

Contract scope: In-memory failure injection at the real refresh write boundary. No actual disk corruption or live HTTP occurred.

Executed output:

```text
refresh OSError injected disk/write failure
offline_replay JSONDecodeError Unterminated string starting at: line 1 column 2 (char 1) cache_remaining= '{"v'
```

Bounded repair: Write and validate a temporary sibling, then atomically replace the cache only after success; remove failed temporary content and retain the original byte digest. Test write/promotion failure, not only failed HTTP responses.

## Reproduce and validate

Every exact executable Python repro and its captured stdout are in `CONTRACT_REVIEW_INITIAL_FINDINGS.json`. From this audit worktree, replay all findings without network or real cache/output writes:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
@'
import json
from pathlib import Path
report = json.loads(Path('docs/v2/final_audit/CONTRACT_REVIEW_INITIAL_FINDINGS.json').read_text(encoding='utf-8'))
for finding in report['findings']:
    print(finding['id'], finding['title'])
    exec(compile(finding['reproduction_python'], '<' + finding['id'] + '>', 'exec'), {})
'@ | & 'C:\Codex\LitDataMatcher-v2\env\Scripts\python.exe' -
```

Existing focused regression executed successfully:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'C:\Codex\LitDataMatcher-v2\env\Scripts\python.exe' -m pytest -p no:cacheprovider -q tests/test_expert_review.py tests/test_calibration_readiness.py tests/test_scientific_dossier.py tests/test_literature_integrity.py tests/test_phase2_safety_invariants.py
```

## Explicit non-findings and limits

- Original acceptance explicitly allows true expert calibration to remain pending (08:64 and 01:95). Actual human labels are not a machine software acceptance gate.
- Expert packet/scorecard, dossier, recovery_contract and provenance_dag helpers are standalone in the searched implementation; their lack of individual CLI commands is not automatically a missing mandatory gate.
- scientific_v2.compile_evidence and assess_requirements are used by the v2 analyze path; source search lifecycle consolidation is used by the optional-adapter CLI.
- Catalog.upsert uses complete payload digests and dependency invalidation; CA-06 is confined to the literature helper.
- The scale receipt explicitly measures a small synthetic fixture and close/reopen recovery; it makes no production-scale claim. No separate scale blocker was proven in this bounded audit.

- No full-suite rerun, sealed holdout, live acquisition, fresh model inference, clean installation, release package validation or scientific expert adjudication was performed by this audit.
- Reproduction mocks intercept all HTTP and file writes; only this report and its JSON are written.
- Findings are initial defects, not repaired-state review; lead must adjudicate/repair/revalidate before closeout.

## Handoff

Lead should preserve this initial evidence, make narrow source/test corrections, and record repair verification separately. Repair agents must not reinterpret these initial failures as repaired-state findings or use real scientific reruns to prove routine helper corrections.
