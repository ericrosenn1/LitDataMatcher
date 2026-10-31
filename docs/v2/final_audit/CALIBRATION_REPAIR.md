# Review and calibration contract repair

Status: **PASS_WITH_LIMITATIONS** for this bounded engineering repair. Real expert validation/calibration remains pending. No real expert labels, scientific evaluation reruns, acquisitions, or protected-alpha/holdout changes occurred.

The initial audit proved that adjudication accepted unsupported/duplicate decisions (CA-02), malformed label data produced `CALIBRATED` (CA-03), and acceptance promoted `label_origins=["expert"]` without any fitted calibration result. Eleven regression cases failed against the initial source before repair. The repaired implementation passes 60 focused/integration tests with no failures, errors or skips. New-module Ruff checks and `git diff --check` pass.

## Corrected behavior

- Review packet imports check the version and content identity. Blinded items are sorted independently of incoming ranking order. Null identities and categorical aliases of duplicate reviewer/item/dimension labels are rejected.
- Adjudication revalidates imported labels and accepts only a unique pending target, supported decision, policy/adjudicator identity and rationale. Duplicate targets are rejected as a group. Accepted decisions retain packet/source provenance and adjudicator identity; their existence never claims fitted calibration or verified expert credentials.
- Scorecard v2 validates status, categorical dimension, provenance, split family, observation identity, binary labels and finite scores in `[0, 1]`. Valid fixed-threshold metrics have `DESCRIPTIVE_ONLY` status. The scorecard always retains `NOT_CALIBRATED` or `PENDING_EXPERT_REVIEW`; `DESCRIPTIVE_LABEL_QA_READY` is a separate readiness field. The former test equating two-class accuracy with calibration was corrected as invalid science.
- Optional `fit_validate_calibration` actually fits a monotonic isotonic transform with group weights, retains a versioned model, and evaluates separate groups with a proper scoring rule. It uses caller-declared criteria; it does not choose scientific endpoints, label conversions or acceptance thresholds. `verify_calibration_result` recomputes the retained fit and validation to detect altered labels, model parameters, status or hashes.
- Acceptance validator v3 requires a hashed, executed G10 `label_provenance` reference and a replay-verifiable real expert result with the matching protocol, validation/transfer role and no tuning exposure. Expert label origin alone leaves calibration pending. The supported legacy threshold-tuning CLI remains an exploratory utility; its report does not satisfy this calibration gate.

## Versioned calibration input contract

Call `fit_validate_calibration(records, protocol=protocol)` only after the scientific owner supplies a fixed protocol and legitimate labels. These are importable Python APIs; no separate command is needed for the bounded repair.

Every retained record requires:

| Field | Meaning |
| --- | --- |
| `record_id`, `record_status` | Unique observation identity and `RETAINED` status |
| `label_origin` | `source_determined` or `expert`; neither is inferred |
| `label_provenance` | Source locator and the protocol's `label_definition`; expert records additionally require reviewer identity and `method="human_review"` |
| `label`, `score` | An explicit binary integer outcome and finite score in `[0, 1]` |
| `dimension` | Relevance, question validity, dataset compatibility or answerability; the protocol must define its binary interpretation |
| `split_family`, `split_role`, `group_id` | Declared family, `FIT`/`VALIDATION`, and study/cohort grouping identity; fit and validation group sets must be disjoint |
| `split_provenance` | Source locator for the declared partition/group assignment |
| `score_definition`, `ablation` | Declared scoring output and one ablation; the fit never pools unrelated outcomes or ablations |
| `data_origin` | Explicit `real` or `synthetic_fixture`; mixed origins are rejected |

The supplied protocol requires `protocol_id`, `source_locator`, `split_family`, `dimension`, `label_definition`, `score_definition`, `method="isotonic_pava_step_v1"`, `predeclared=true`, `data_origin`, `minimum_records_per_split`, `minimum_groups_per_split`, `maximum_brier_score`, and `minimum_brier_improvement`. Structural minimums require at least two groups/records and both classes in each split; meaningful scientific floors must be supplied in the protocol. No default quality thresholds are supplied.

`partition_digest` fingerprints the sorted list of `{record_id, group_id, split_role}` objects with `litdatamatcher.data_plane.digest`. The API refuses different partitions. Group identities are normalized for surrounding whitespace so spelling variants cannot conceal overlap.

The fit pools tied scores and adjacent violations. Each group has equal total fitting weight. Prediction uses the documented monotone step transform and boundary blocks outside the fitted score range. Validation uses equal-group-weighted Brier scores against the retained binary outcomes. Its labels never affect the model fit.

| Result | Scope |
| --- | --- |
| `FITTED_NOT_VALIDATED` | Model fitted, declared held-out proper-score checks failed |
| `SYNTHETIC_VALIDATION_ONLY` | Synthetic computational checks passed; no scientific calibration claim |
| `SOURCE_CALIBRATION_VALIDATED_IN_SCOPE` | Declared real source-label fit passed its supplied protocol; not expert validation |
| `EXPERT_CALIBRATED_IN_SCOPE` | Declared real human-review labels and fitted validation pass the supplied scoped protocol; acceptance additionally verifies the executed/hashed evidence envelope |

## Executed evidence

Receipt directory: `C:\Codex\LitDataMatcher-v2\data\final_campaign_20260913\calibration_repair`.

- `red_before.json`, `red_before.xml`, `red_before.log`: eleven expected failures before source correction.
- `green_after.json`, `green_after.xml`, `green_command_*.log`: exact commands, source file hashes, final source checkpoint, and 60 passing focused/integration tests.
- `scorecard_receipt_v2.json`: descriptive scorecard readiness plus pending-expert checks.
- `synthetic_fit_v1.json`: complete declared synthetic fit, retained inputs/protocol/model, independent validation predictions and replay verification; status `SYNTHETIC_VALIDATION_ONLY`.
- `MANIFEST.json`: hashes and sizes of the repair evidence files.

The regression checks actual fitting against a known pooled isotonic solution, input-order determinism, validation-label separation, false calibration status, input/model/hash tampering, group overlap, missing provenance, changed partitions, classes/denominators, invalid numerics, and the acceptance evidence envelope. Acceptance wiring tests mock only the separately tested real-result verifier when a positive envelope case is needed; they do not fabricate real expert labels.

## Limits and integration

The software can validate recorded identities, provenance shape, split membership, hashes, numerical fit and explicit criteria. It cannot authenticate expert credentials, prove that externally assigned groups are biologically independent, or establish that a supplied protocol was scientifically justified and fixed in advance. Those remain external evidence obligations. Merely declaring `predeclared=true` is not independent proof of preregistration.

This new API is opt-in machinery. Existing ranking/holdout thresholds and scientific endpoints are unchanged. No real calibration is claimed in this handoff. The old v1 calibration-readiness receipt is retained historically and superseded as a scientific calibration claim.

The lead should integrate this scoped commit and run the assembled final regression/review. Full-suite release validation, packaging and campaign completion remain lead-owned. The initial eight-finding audit is preserved unchanged; this document adjudicates only CA-02/03 and the acceptance calibration overclaim.
