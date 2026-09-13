# V2.4 calibration readiness

The corrected `build_calibration_scorecard` emits schema `v2_4_calibration_scorecard_v2`. Valid categorical-label provenance, a nonblank split family, unique observation identities, finite bounded scores and binary labels permit **descriptive QA only**. Its `calibration_status` remains `NOT_CALIBRATED`; `readiness_status=DESCRIPTIVE_LABEL_QA_READY` identifies usable descriptive metrics. Pending labels retain `PENDING_EXPERT_REVIEW`. Invalid rows/denominators suppress metrics. Novelty, unresolvedness, and scientific significance are never calibrated.

`fit_validate_calibration` is a separate opt-in, versioned isotonic calibration API. It requires an explicitly supplied protocol/method, fixed label and score definitions, recorded provenance, a fingerprinted partition, disjoint fit/validation groups, both classes and caller-declared sample/group floors and proper-score criteria. It retains inputs, fitted model, validation predictions and hashes. `verify_calibration_result` recomputes the result. Synthetic fixtures remain `SYNTHETIC_VALIDATION_ONLY`; a fitted model failing its declared validation remains `FITTED_NOT_VALIDATED`.

Acceptance no longer promotes an `expert` label-origin flag into `EXPERT_CALIBRATED`. It requires an executed G10 reference to a currently hashed, replay-verifiable real expert calibration result under the matching validation protocol. Product acceptance may still close with expert calibration pending.

Repair validation: 60 focused/integration tests passed, no failures/errors/skips. Exact red-before/green-after commands, XML/logs, source hashes and a declared synthetic fit are under `C:\Codex\LitDataMatcher-v2\data\final_campaign_20260913\calibration_repair`. See `docs/v2/final_audit/CALIBRATION_REPAIR.md` for input contracts and limits.

The earlier v1 receipt at `data\phase2\v2_4_calibration_readiness\receipt.json` (SHA-256 `14ac6ece9b6e5b85fd91c7e4275143bbba01313863d01b9a7c9235411690c1ab`) is preserved as historical evidence. Its two-class fixture accuracy did not establish scientific calibration and must not be used as a current calibration claim.
